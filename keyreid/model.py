"""KeyRe-ID (Section 3 of the paper).

A keypoint-guided Transformer framework pairing a global identity branch with a
pose-adaptive local branch on a shared ViT-B/16 encoder.

Global branch (Section 3.3)
    A dedicated Transformer encoder block refines the frame tokens (Eq. 6). A temporal
    attention module (2D convolution followed by a 1D temporal convolution, non-negative
    scores, softmax over frames) weights the frame-wise [CLS] tokens, whose weighted sum is
    the clip-level representation f_global (Eq. 7). The prediction head (batch normalization
    followed by a fully connected layer) gives the embedding e_global and the logits y_global.

Local branch: Keypoint-guided Part Selection (KPS, Section 3.4)
    Clip-level patch descriptors   C(j,:) = P_j^(1) || ... || P_j^(T)                     (Eq. 8)
    Internal part-support head     p_r(j) = max(0, w_r^T LN(C_bar(j,:)) + b_r)            (Eq. 9)
        one rectified output per anatomical region (R = 6), 6.2K parameters; trained with
        the soft Dice region-overlap objective L_part against reference part maps (Eq. 10).
    Part merge                     s_k(j) = max_{r in R_k} p_r(j) / max(max_j' p_r(j'), eps) (Eq. 11)
    Part readout                   f_k = [ B([q_k; C]; S_k) ]_0,  S_k = { j : s_k(j) > 0 }  (Eq. 12)
        Each part query uses a binary attention mask defined by its positive support:
        patches outside S_k are excluded from the key and value set of the query. If the
        support is empty, attention is left unrestricted.

Neither a pose estimator nor any keypoint map is required at inference.
"""
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TransReID, Block, resize_pos_embed

NUM_REGIONS = 6  # R: head, torso, left arm, right arm, left leg, right leg (Table 1)
PART_GROUPS = {  # R_k: regions merged into each of the K parts
    3: [[0], [1, 2, 3], [4, 5]],       # head, torso + arms, legs
    4: [[0], [1], [2, 3], [4, 5]],     # head, torso, arms, legs (default)
    6: [[0], [1], [2], [3], [4], [5]],
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class KeyReID(nn.Module):
    def __init__(self, num_classes, camera_num, num_parts=4, cam_lambda=3.0, pretrained=None):
        super().__init__()
        assert num_parts in PART_GROUPS, f"num_parts must be one of {sorted(PART_GROUPS)}"
        self.in_planes = 768
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.groups = PART_GROUPS[num_parts]

        # ---------------- shared ViT encoder (Section 3.2, Eq. 1) ----------------
        self.base = TransReID(
            img_size=[256, 128], patch_size=16, stride_size=[16, 16], embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True, camera=camera_num, drop_path_rate=0.1,
            drop_rate=0.0, attn_drop_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            cam_lambda=cam_lambda)
        if pretrained:
            self.base.load_param(torch.load(pretrained, map_location='cpu'), load=True)

        # ---------------- global branch (Section 3.3) ----------------
        self.b1 = nn.Sequential(copy.deepcopy(self.base.blocks[-1]), copy.deepcopy(self.base.norm))
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # ---------------- local branch: shared block B and part prediction heads (Section 3.4) ----------------
        self.part_dim = 3072  # T * D with T = 4 frames per clip
        self.b2 = nn.Sequential(
            Block(dim=self.part_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                  drop=0, attn_drop=0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)),
            nn.LayerNorm(self.part_dim))
        for k in range(1, num_parts + 1):
            bn = nn.BatchNorm1d(self.part_dim)
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
            setattr(self, f'bottleneck_{k}', bn)
        for k in range(1, num_parts + 1):
            cls = nn.Linear(self.part_dim, num_classes, bias=False)
            cls.apply(weights_init_classifier)
            setattr(self, f'classifier_{k}', cls)

        # ---------------- temporal attention module (Section 3.3) ----------------
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1, 1])
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming)
        self.attention_tconv.apply(weights_init_kaiming)

        # ---------------- internal part-support head (Eq. 9) and learnable part queries q_k ----------------
        self.kp_head = nn.Sequential(nn.LayerNorm(self.in_planes), nn.Linear(self.in_planes, NUM_REGIONS))
        self.part_query = nn.Parameter(torch.randn(num_parts, self.part_dim) * 0.02)

    # ------------------------------------------------------------------
    @property
    def part_bottlenecks(self):
        return [getattr(self, f'bottleneck_{k}') for k in range(1, self.num_parts + 1)]

    @property
    def part_classifiers(self):
        return [getattr(self, f'classifier_{k}') for k in range(1, self.num_parts + 1)]

    def _restricted_readout(self, query, tokens, outside):
        """Part readout of Eq. (12): shared block B over [q_k; C] with a binary attention mask for the query.

        query   : [B, 1, P]
        tokens  : [B, n, P]
        outside : [B, n] bool, True for patches outside the part support
        """
        blk, norm = self.b2[0], self.b2[1]
        x = torch.cat((query, tokens), dim=1)
        h = blk.norm1(x)
        attn = blk.attn
        B, N, C = h.shape
        qkv = attn.qkv(h).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logit = (q @ k.transpose(-2, -1)) * attn.scale
        if outside.any():
            blocked = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            blocked[:, 1:] = outside
            blocked[outside.all(dim=1), :] = False      # empty support: unrestricted attention
            row = logit[:, :, 0, :].masked_fill(blocked[:, None, :], float('-inf'))
            logit = torch.cat([row.unsqueeze(2), logit[:, :, 1:, :]], dim=2)
        a = attn.attn_drop(logit.softmax(dim=-1))
        out = attn.proj_drop(attn.proj((a @ v).transpose(1, 2).reshape(B, N, C)))
        x = x + blk.drop_path(out)
        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
        return norm(x)[:, 0]

    @staticmethod
    def _pool_reference_maps(ref_maps, b, t):
        """Reference part maps [B, T, 6, 256, 128] -> average-pooled onto the 16x8 patch grid, averaged over T (Eq. 5)."""
        hp = F.unfold(ref_maps.view(b * t, NUM_REGIONS, 256, 128), kernel_size=16, stride=16)
        hp = hp.view(b * t, NUM_REGIONS, 16 * 16, 128).mean(dim=2)
        return hp.transpose(1, 2).view(b, t, 128, NUM_REGIONS).mean(dim=1)

    @staticmethod
    def part_support_loss(pred_support, target):
        """Soft Dice region-overlap objective L_part of Eq. (10); inputs are [B, R, n]."""
        num = 2.0 * (pred_support * target).sum(-1)
        den = (pred_support * pred_support).sum(-1) + (target * target).sum(-1) + 1e-6
        return (1.0 - num / den).mean()

    # ------------------------------------------------------------------
    def forward(self, x, cam_label, ref_maps=None):
        """x: [B, T, 3, 256, 128] clips; cam_label: camera index of every frame (B*T).

        Training returns (identity logits, features, non-negative temporal attention scores a_global,
        L_part). Evaluation returns the clip embeddings [e_global, e_1/K, ..., e_K/K], from which
        the retrieval descriptor of Eq. (17) is formed (keyreid/evaluation.py).
        """
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        features = self.base(x, cam_label=cam_label)

        # ---------------- global branch: Eqs. (6)-(7) ----------------
        b1_feat = self.b1(features)
        gf = b1_feat[:, 0].unsqueeze(2).unsqueeze(3)
        a = F.relu(self.attention_conv(gf)).view(b, t, self.middle_dim).permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a)).view(b, t)
        attn_scores = a
        a = F.softmax(a, dim=1).unsqueeze(-1).expand(b, t, self.in_planes)
        global_feat = torch.sum(torch.mul(gf.view(b, t, -1), a), 1).view(b, self.in_planes)
        feat = self.bottleneck(global_feat)

        # ---------------- clip-level patch descriptors C (Eq. 8) ----------------
        n_tok, dim = features.size(1), features.size(2)
        f = features.view(b, t, n_tok, dim).permute(0, 2, 1, 3).reshape(b, n_tok, t * dim)
        patch_feats = f[:, 1:]

        # ---------------- internal part-support prediction p_r (Eq. 9) ----------------
        pred = self.kp_head(f[:, 1:].view(b, n_tok - 1, t, dim).mean(2))
        p = F.relu(pred.float().transpose(1, 2))                        # [B, R, n]; the rectifier allows exact zeros
        part_loss = None
        if ref_maps is not None:
            target = self._pool_reference_maps(ref_maps, b, t)
            # reference support m_r (Eq. 5), rescaled so that its largest entry is one
            g = ((target.float() + 1.0) / 2.0).clamp_min(0.0).transpose(1, 2)
            g = g / g.amax(-1, keepdim=True).clamp_min(1e-8)
            part_loss = self.part_support_loss(p, g)

        # per-region normalisation and merge into K parts (Eq. 11), in the signed range of the reference maps
        w = p / p.amax(-1, keepdim=True).clamp_min(1e-8)
        region_support = (w * 2.0 - 1.0).transpose(1, 2).to(pred.dtype)     # [B, n, 6]
        part_support = torch.stack([region_support[:, :, gi].max(dim=-1).values
                                    for gi in self.groups], dim=-1)      # [B, n, K]

        # ---------------- part readout with restricted attention (Eq. 12) ----------------
        part_f = []
        for i in range(self.num_parts):
            s = ((part_support[:, :, i].unsqueeze(-1) + 1.0) / 2.0).clamp_min(0.0).float()
            tokens = patch_feats * s.to(patch_feats.dtype)
            outside = s.squeeze(-1) <= 0
            q = self.part_query[i].to(tokens.dtype).view(1, 1, -1).expand(b, 1, -1)
            part_f.append(self._restricted_readout(q, tokens, outside))

        part_bn = [bn(pf) for bn, pf in zip(self.part_bottlenecks, part_f)]
        if self.training:
            ids = [self.classifier(feat)] + [c(pb) for c, pb in zip(self.part_classifiers, part_bn)]
            return ids, [global_feat] + part_f, attn_scores, part_loss
        return torch.cat([feat] + [pb / self.num_parts for pb in part_bn], dim=1)

    # ------------------------------------------------------------------
    def load_pretrained_vit(self, path):
        """Load ImageNet-21K ViT-B/16 weights into the backbone (position embedding resized to 16x8)."""
        param_dict = torch.load(path, map_location='cpu')
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
                v = resize_pos_embed(v, self.base.pos_embed, self.base.patch_embed.num_y,
                                     self.base.patch_embed.num_x)
            new_k = k
            if k.startswith('base.') and k[5:] in model_dict:
                new_k = k[5:]
            elif not k.startswith('base.') and ('base.' + k) in model_dict:
                new_k = 'base.' + k
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                new_param_dict[new_k] = v
        model_dict.update(new_param_dict)
        self.load_state_dict(model_dict, strict=False)

    def load_finetune_init(self, path, keep_camera_embedding):
        """Initialise from a trained KeyRe-ID model (identity classifiers are not transferred).

        keep_camera_embedding: copy the first rows of the camera embedding (PRID-2011);
        otherwise the camera embedding keeps its fresh initialisation (iLIDS-VID).
        """
        state = torch.load(path, map_location='cpu')
        own = self.state_dict()
        new = {}
        for k, v in state.items():
            if 'classifier' in k:
                continue
            if k == 'base.Cam':
                if not keep_camera_embedding:
                    continue
                exp = own[k].shape
                if v.shape[0] >= exp[0]:
                    v = v[:exp[0]]
                else:
                    nv = torch.randn(exp)
                    nv[:v.shape[0]] = v
                    v = nv
            if k in own and own[k].shape == v.shape:
                new[k] = v
        own.update(new)
        self.load_state_dict(own, strict=False)


def load_checkpoint(model, path):
    """Load released KeyRe-ID weights; identity classifiers are not needed for retrieval."""
    state = torch.load(path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if 'classifier' not in k]
    unexpected = [k for k in unexpected if 'classifier' not in k]
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing={missing} unexpected={unexpected}')
    return model
