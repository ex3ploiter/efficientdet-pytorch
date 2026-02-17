import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# --- 1. Helper: Feature Projector ---
class FeatureProjector(nn.Module):
    def __init__(self, in_channels_list, embed_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1) 
            for in_ch in in_channels_list
        ])
        
    def forward(self, features):
        projected = []
        for feat, proj in zip(features, self.projections):
            # Safe permute: [B, H, W, C] -> [B, C, H, W]
            if feat.shape[-1] == proj.in_channels:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            x = proj(feat)
            projected.append(x)
        return projected

# --- 2. The Core: Dynamic Scale Router ---
class DynamicScaleRouter(nn.Module):
    def __init__(self, num_queries, embed_dim, num_scales=4):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.scout_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.scout_norm = nn.LayerNorm(embed_dim)
        self.router_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_scales)
        )

    def forward(self, spatial_features):
        # Flatten deepest feature for context
        global_feat = spatial_features[-1].flatten(2).transpose(1, 2)
        B = global_feat.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) 

        # Scout & Route
        scout_out, _ = self.scout_attn(queries, global_feat, global_feat)
        queries = self.scout_norm(queries + scout_out)
        logits = self.router_mlp(queries) 
        
        # FIX FOR NaN: Use plain Softmax initially. Gumbel can be unstable.
        # We can switch back to Gumbel later, but Softmax is safer for debugging.
        routing_map = F.softmax(logits, dim=-1) 
        return queries, routing_map

# --- 3. The Head: NSA ---
class NSAHead(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, queries, spatial_features, routing_map):
        output_queries = torch.zeros_like(queries)
        num_scales = len(spatial_features)
        
        for k in range(num_scales):
            # Softmax routing means routing_map is continuous (0.0 to 1.0), not boolean.
            # We weight the attention output by this probability.
            prob_k = routing_map[:, :, k].unsqueeze(-1) # [B, Q, 1]
            
            # Optimization: Skip if weight is negligible
            if prob_k.max() < 1e-4: continue

            target_feat = spatial_features[k].flatten(2).transpose(1, 2)
            attn_out, _ = self.attn(queries, target_feat, target_feat)
            
            output_queries += attn_out * prob_k

        queries = self.norm(queries + output_queries)
        queries = queries + self.ffn(queries)
        return queries

# --- 4. Main Model (Stabilized) ---
class EfficientDet(nn.Module):
    def __init__(self, config, pretrained_backbone=False, alternate_init=False):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.embed_dim = 256 
        self.num_anchors = 9
        
        # A. Backbone
        self.backbone = create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            features_only=True,
            out_indices=(1, 2, 3) 
        )
        for param in self.backbone.parameters():
          param.requires_grad = False
        
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs'])
        channels_list = [info['num_chs'] for info in feature_info]
        
        # B. Projector & Smoother
        self.projector = FeatureProjector(channels_list, self.embed_dim)
        # New layer to smooth out interpolation artifacts
        self.smoother = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)
        
        # C. Components
        self.router = DynamicScaleRouter(300, self.embed_dim, len(channels_list))
        self.nsa_head = NSAHead(self.embed_dim)
        
        # D. FPN Layers
        self.conv_p6 = nn.Conv2d(self.embed_dim, self.embed_dim, 3, stride=2, padding=1)
        self.conv_p7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, stride=2, padding=1)
        )

        # E. Heads
        self.class_head = nn.Conv2d(self.embed_dim, self.num_anchors * self.num_classes, 3, padding=1)
        self.bbox_head = nn.Conv2d(self.embed_dim, self.num_anchors * 4, 3, padding=1)
        
        # --- CRITICAL FIX: Initialization to prevent NaNs ---
        self._init_weights()

    def _init_weights(self):
        # 1. Initialize Class Head Bias (RetinaNet Trick)
        # This forces the network to predict "Background" with high confidence initially.
        # Prior probability pi = 0.01
        pi = 0.01
        bias_value = -math.log((1 - pi) / pi)
        
        nn.init.constant_(self.class_head.bias, bias_value)
        nn.init.normal_(self.class_head.weight, std=0.01)
        
        # 2. Initialize Box Head with very small numbers
        nn.init.constant_(self.bbox_head.bias, 0)
        nn.init.normal_(self.bbox_head.weight, std=0.001)

        # 3. Init other layers
        for m in [self.conv_p6, self.conv_p7, self.smoother]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        original_h, original_w = x.shape[-2:]

        # 1. DOWNSIZE (224 for Swin)
        x_timm = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x_timm) 
        
        # 2. Project
        proj_features = self.projector(features)
        
        # 3. UPSAMPLE & SMOOTH (Critical Step)
        restored_features = []
        strides = [8, 16, 32] 
        
        for i, feat in enumerate(proj_features):
            target_h = original_h // strides[i]
            target_w = original_w // strides[i]
            
            # Interpolate
            upsampled = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # Smooth the artifacts
            smoothed = self.smoother(upsampled)
            restored_features.append(smoothed)
            
        # 4. Sparse Logic
        queries, routing_map = self.router(restored_features)
        refined_queries = self.nsa_head(queries, restored_features, routing_map)
        
        # 5. Context Injection
        global_context = refined_queries.mean(dim=1).view(-1, self.embed_dim, 1, 1)
        enhanced_features = [f + global_context for f in restored_features]
        
        # 6. Build Pyramid
        p3, p4, p5 = enhanced_features
        p6 = self.conv_p6(p5)
        p7 = self.conv_p7(p6)
        pyramid = [p3, p4, p5, p6, p7]
        
        # 7. Prediction
        cls_outputs = []
        box_outputs = []
        for feat in pyramid:
            cls_outputs.append(self.class_head(feat))
            box_outputs.append(self.bbox_head(feat))

        return cls_outputs, box_outputs