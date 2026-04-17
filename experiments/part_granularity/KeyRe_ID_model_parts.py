"""
KeyRe-ID model with configurable number of parts.
Supports: 3, 4, 6 parts by merging existing 6-channel heatmaps.
"""
import torch
import torch.nn as nn
import copy
from vit_ID import TransReID, Block
from functools import partial
from torch.nn import functional as F
from vit_ID import resize_pos_embed
from KeyRe_ID_model import TCSS, weights_init_kaiming, weights_init_classifier


# Part definitions: which heatmap channels to merge
PART_CONFIGS = {
    3: {
        'names': ['Upper', 'Arms', 'Legs'],
        'channels': [[0, 1], [2, 3], [4, 5]],  # head+torso, L+R arm, L+R leg
    },
    4: {
        'names': ['Head', 'Torso', 'Arms', 'Legs'],
        'channels': [[0], [1], [2, 3], [4, 5]],  # head, torso, arms merged, legs merged
    },
    6: {
        'names': ['Head', 'Torso', 'L-Arm', 'R-Arm', 'L-Leg', 'R-Leg'],
        'channels': [[0], [1], [2], [3], [4], [5]],  # original
    },
}


class KeyRe_ID_Parts(nn.Module):
    def __init__(self, num_classes, camera_num, pretrainpath, num_parts=6):
        super(KeyRe_ID_Parts, self).__init__()
        assert num_parts in PART_CONFIGS, f"num_parts must be one of {list(PART_CONFIGS.keys())}"
        
        self.in_planes = 768
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.part_config = PART_CONFIGS[num_parts]
        
        print(f"[Part Config] num_parts={num_parts}: {self.part_config['names']}")
        
        self.base = TransReID(
            img_size=[256, 128], patch_size=16, stride_size=[16, 16],
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            camera=camera_num, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), cam_lambda=3.0)
        
        if pretrainpath:
            state_dict = torch.load(pretrainpath, map_location='cpu')
            self.base.load_param(state_dict, load=True)
        
        # Global Branch
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        # Local Branch - dynamic number of parts
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.block1 = Block(
            dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0, attn_drop=0, drop_path=dpr[11],
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.b2 = nn.Sequential(self.block1, nn.LayerNorm(3072))
        
        # Create bottlenecks and classifiers for each part
        self.part_bottlenecks = nn.ModuleList()
        self.part_classifiers = nn.ModuleList()
        for i in range(num_parts):
            bn = nn.BatchNorm1d(3072)
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
            self.part_bottlenecks.append(bn)
            
            cls = nn.Linear(3072, self.num_classes, bias=False)
            cls.apply(weights_init_classifier)
            self.part_classifiers.append(cls)
        
        # Video attention
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1, 1])
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming)
        self.attention_tconv.apply(weights_init_kaiming)
        
        self.shift_num = 5
        self.part = num_parts
    
    def _merge_heatmap_channels(self, heatmap_weights):
        """Merge 6-channel heatmap weights into num_parts channels."""
        # heatmap_weights: [B, 128, 6]
        merged = []
        for ch_indices in self.part_config['channels']:
            if len(ch_indices) == 1:
                merged.append(heatmap_weights[:, :, ch_indices[0]])
            else:
                # Take max across channels to merge
                merged.append(torch.stack([heatmap_weights[:, :, c] for c in ch_indices], dim=-1).max(dim=-1)[0])
        return torch.stack(merged, dim=-1)  # [B, 128, num_parts]
    
    def forward(self, x, heatmaps, label=None, cam_label=None, view_label=None):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        features = self.base(x, cam_label=cam_label)
        
        # Global Branch
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]
        global_feat = global_feat.unsqueeze(dim=2).unsqueeze(dim=3)
        a = F.relu(self.attention_conv(global_feat))
        a = a.view(b, t, self.middle_dim).permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a)).view(b, t)
        a_vals = a
        a = F.softmax(a, dim=1)
        x_g = global_feat.view(b, t, -1)
        a_exp = a.unsqueeze(-1).expand(b, t, self.in_planes)
        att_x = torch.mul(x_g, a_exp).sum(1)
        global_feat = att_x.view(b, self.in_planes)
        feat = self.bottleneck(global_feat)
        
        # Local Branch - Heatmap Processing
        heatmaps_proc = heatmaps.view(b * t, 6, 256, 128)
        heatmap_patches = F.unfold(heatmaps_proc, kernel_size=16, stride=16)
        heatmap_patches = heatmap_patches.view(b * t, 6, 16 * 16, 128).mean(dim=2)
        heatmap_weights = heatmap_patches.transpose(1, 2)  # [B*T, 128, 6]
        heatmap_weights = heatmap_weights.view(b, t, 128, 6).mean(dim=1)  # [B, 128, 6]
        
        # Merge channels based on part config
        heatmap_weights = self._merge_heatmap_channels(heatmap_weights)  # [B, 128, num_parts]
        
        # TCSS
        x_l, token = TCSS(features, self.shift_num, b, t)
        patch_feats = x_l
        
        # Process each part
        part_feats = []
        part_bns = []
        for i in range(self.num_parts):
            weight = heatmap_weights[:, :, i].unsqueeze(-1)
            part = patch_feats * weight
            part = self.b2(torch.cat((token, part), dim=1))
            part_f = part[:, 0]
            part_feats.append(part_f)
            part_bns.append(self.part_bottlenecks[i](part_f))
        
        if self.training:
            Global_ID = self.classifier(feat)
            Local_IDs = [self.part_classifiers[i](part_bns[i]) for i in range(self.num_parts)]
            return [Global_ID] + Local_IDs, [global_feat] + part_feats, a_vals
        else:
            parts_concat = torch.cat([bn / self.part for bn in part_bns], dim=1)
            return torch.cat([feat, parts_concat], dim=1)

    def load_param(self, trained_path, load=False):
        print("Run load_param (Parts model)")
        if not load:
            param_dict = torch.load(trained_path, map_location='cpu')
        else:
            param_dict = trained_path
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        model_dict = self.state_dict()
        new_param_dict = {}
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.base.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.base.pos_embed.shape:
                v = resize_pos_embed(v, self.base.pos_embed, self.base.patch_embed.num_y, self.base.patch_embed.num_x)
            
            new_k = k
            if k.startswith("base.") and k[5:] in model_dict:
                new_k = k[5:]
            elif not k.startswith("base.") and ("base." + k) in model_dict:
                new_k = "base." + k

            if new_k in ['Cam', 'base.Cam'] and new_k in model_dict:
                expected_shape = model_dict[new_k].shape
                if v.shape[0] > expected_shape[0]:
                    v = v[:expected_shape[0], :, :]
                elif v.shape[0] < expected_shape[0]:
                    new_v = torch.randn(expected_shape)
                    new_v[:v.shape[0], :, :] = v
                    v = new_v
                new_param_dict[new_k] = v
                continue

            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                new_param_dict[new_k] = v

        model_dict.update(new_param_dict)
        self.load_state_dict(model_dict, strict=False)
        print("Checkpoint loaded (strict=False).")
