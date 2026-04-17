"""
KeyRe-ID model with ablation flags.
Supports: use_global, use_local, use_tcss, use_kps
"""
import torch
import torch.nn as nn
import copy
from vit_backbone import TransReID, Block
from functools import partial
from torch.nn import functional as F
from vit_backbone import resize_pos_embed
from keyreid import TCSS, weights_init_kaiming, weights_init_classifier, KeyReID


class KeyReIDAblation(KeyReID):
    """KeyReID with ablation support. Inherits all weights from KeyReID."""

    def __init__(self, num_classes, camera_num, pretrainpath,
                 use_global=True, use_local=True, use_tcss=True, use_kps=True):
        super().__init__(num_classes, camera_num, pretrainpath)
        self.use_global = use_global
        self.use_local = use_local
        self.use_tcss = use_tcss
        self.use_kps = use_kps
        print(f"[Ablation] global={use_global}, local={use_local}, tcss={use_tcss}, kps={use_kps}")

    def forward(self, x, heatmaps, label=None, cam_label=None, view_label=None):
        b = x.size(0)
        t = x.size(1)
        x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        features = self.base(x, cam_label=cam_label)

        # ------- Global Branch -------
        if self.use_global:
            b1_feat = self.b1(features)
            global_feat = b1_feat[:, 0]
            global_feat = global_feat.unsqueeze(dim=2).unsqueeze(dim=3)
            a = F.relu(self.attention_conv(global_feat))
            a = a.view(b, t, self.middle_dim)
            a = a.permute(0,2,1)
            a = F.relu(self.attention_tconv(a))
            a = a.view(b, t)
            a_vals = a
            a = F.softmax(a, dim=1)
            x_g = global_feat.view(b, t, -1)
            a_exp = torch.unsqueeze(a, -1).expand(b, t, self.in_planes)
            att_x = torch.mul(x_g, a_exp)
            att_x = torch.sum(att_x, 1)
            global_feat = att_x.view(b, self.in_planes)
            feat = self.bottleneck(global_feat)
        else:
            # Dummy attention values for loss computation
            a_vals = torch.zeros(b, t, device=x.device)
            feat = None
            global_feat = None

        # ------- Local Branch -------
        if self.use_local:
            # Heatmap Processing
            heatmaps_proc = heatmaps.view(b*t, 6, 256, 128)
            heatmap_patches = F.unfold(heatmaps_proc, kernel_size=16, stride=16)
            heatmap_patches = heatmap_patches.view(b*t, 6, 16*16, 128).mean(dim=2)
            heatmap_weights = heatmap_patches.transpose(1, 2)
            heatmap_weights = heatmap_weights.view(b, t, 128, 6).mean(dim=1)

            # TCSS or simple concat
            if self.use_tcss:
                x_l, token = TCSS(features, self.shift_num, b, t)
            else:
                # Simple temporal concat without shift/shuffle
                feats_reshaped = features.view(b, features.size(1), t*features.size(2))
                token = feats_reshaped[:, 0:1]
                x_l = feats_reshaped[:, 1:]  # remove cls token

            patch_feats = x_l

            # KPS or uniform weights
            part_feats = []
            for i in range(6):
                if self.use_kps:
                    weight = heatmap_weights[:, :, i].unsqueeze(-1)
                    part = patch_feats * weight
                else:
                    part = patch_feats  # uniform (no weighting)
                part = self.b2(torch.cat((token, part), dim=1))
                part_feats.append(part[:, 0])

            part1_f, part2_f, part3_f, part4_f, part5_f, part6_f = part_feats
            part1_bn = self.bottleneck_1(part1_f)
            part2_bn = self.bottleneck_2(part2_f)
            part3_bn = self.bottleneck_3(part3_f)
            part4_bn = self.bottleneck_4(part4_f)
            part5_bn = self.bottleneck_5(part5_f)
            part6_bn = self.bottleneck_6(part6_f)
        else:
            part1_f = part2_f = part3_f = part4_f = part5_f = part6_f = None
            part1_bn = part2_bn = part3_bn = part4_bn = part5_bn = part6_bn = None

        # ------- Output -------
        if self.training:
            if self.use_global and self.use_local:
                Global_ID = self.classifier(feat)
                Local_IDs = [self.classifier_1(part1_bn), self.classifier_2(part2_bn),
                             self.classifier_3(part3_bn), self.classifier_4(part4_bn),
                             self.classifier_5(part5_bn), self.classifier_6(part6_bn)]
                return [Global_ID] + Local_IDs, \
                       [global_feat, part1_f, part2_f, part3_f, part4_f, part5_f, part6_f], a_vals
            elif self.use_global and not self.use_local:
                Global_ID = self.classifier(feat)
                return [Global_ID], [global_feat], a_vals
            elif not self.use_global and self.use_local:
                Local_IDs = [self.classifier_1(part1_bn), self.classifier_2(part2_bn),
                             self.classifier_3(part3_bn), self.classifier_4(part4_bn),
                             self.classifier_5(part5_bn), self.classifier_6(part6_bn)]
                return Local_IDs, [part1_f, part2_f, part3_f, part4_f, part5_f, part6_f], a_vals
        else:
            parts = []
            if self.use_global:
                parts.append(feat)
            if self.use_local:
                parts.extend([part1_bn/self.part, part2_bn/self.part, part3_bn/self.part,
                              part4_bn/self.part, part5_bn/self.part, part6_bn/self.part])
            return torch.cat(parts, dim=1)
