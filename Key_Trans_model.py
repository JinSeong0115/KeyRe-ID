import torch
import torch.nn as nn
import copy
from vit_ID import TransReID, Block, resize_pos_embed
from functools import partial
from torch.nn import functional as F
from token_fusion import FusionToken
from kpr import PixelToPartClassifier, GlobalAveragePoolingHead

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
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
      
class KeyTransReID(nn.Module):
    def __init__(self, num_classes, camera_num, pretrainpath, parts_num=6):
        super(KeyTransReID, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes
        self.parts_num = parts_num
        self.t = 4  # number of frames
        
        # fusion token
        self.fusion_module = FusionToken(embed_dim=768)
        
        # TransReID backbone
        self.base = TransReID(
            img_size=[256, 128], patch_size=16, stride_size=[16, 16], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            camera=camera_num,  drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), cam_lambda=3.0
        )
        
        # load pretrained weight
        state_dict = torch.load(pretrainpath, map_location='cpu')
        self.base.load_param(state_dict, load=True)
        
        #-------------------Global Branch-------------
        block= self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        #-------------------Local Branch-------------
        self.pixel_classifier = PixelToPartClassifier(self.t*self.in_planes, self.parts_num)  # part classifier now outputs 6 channels
        self.parts_pooling = GlobalAveragePoolingHead(self.t*self.in_planes)
        
        # Transformer block for temporal refinement remains
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]  # stochastic depth decay rule
        self.block1 = Block(
            dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0, attn_drop=0, drop_path=dpr[11], norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.b2 = nn.Sequential(
            self.block1,
            nn.LayerNorm(3072)
        )
        
        # Part-based Transformer Refinement
        self.part_tokens = nn.Parameter(torch.randn(self.parts_num, self.t * self.in_planes))  # Learned part tokens for each of 6 parts
        self.part_transformers = nn.ModuleList([copy.deepcopy(self.b2) for _ in range(self.parts_num)])
        
        self.bottleneck_head = nn.BatchNorm1d(3072)
        self.bottleneck_head.bias.requires_grad_(False)
        self.bottleneck_head.apply(weights_init_kaiming)
        self.bottleneck_torso = nn.BatchNorm1d(3072)
        self.bottleneck_torso.bias.requires_grad_(False)
        self.bottleneck_torso.apply(weights_init_kaiming)
        self.bottleneck_left_arm = nn.BatchNorm1d(3072)
        self.bottleneck_left_arm.bias.requires_grad_(False)
        self.bottleneck_left_arm.apply(weights_init_kaiming)
        self.bottleneck_right_arm = nn.BatchNorm1d(3072)
        self.bottleneck_right_arm.bias.requires_grad_(False)
        self.bottleneck_right_arm.apply(weights_init_kaiming)
        self.bottleneck_left_leg = nn.BatchNorm1d(3072)
        self.bottleneck_left_leg.bias.requires_grad_(False)
        self.bottleneck_left_leg.apply(weights_init_kaiming)
        self.bottleneck_right_leg = nn.BatchNorm1d(3072)
        self.bottleneck_right_leg.bias.requires_grad_(False)
        self.bottleneck_right_leg.apply(weights_init_kaiming)
        
        self.classifier_head = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_head.apply(weights_init_classifier)
        self.classifier_torso = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_torso.apply(weights_init_classifier)
        self.classifier_left_arm = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_left_arm.apply(weights_init_classifier)
        self.classifier_right_arm = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_right_arm.apply(weights_init_classifier)
        self.classifier_left_leg = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_left_leg.apply(weights_init_classifier)
        self.classifier_right_leg = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_right_leg.apply(weights_init_classifier)
        
        #-------------------video attention-------------
        self.middle_dim = 256 # middle layer dimension
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1,1])
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 
        

    def forward(self, fusion_tokens, heatmap, cam_label=None):  # label is unused if self.cos_layer == 'no'
        t = self.t
        b = fusion_tokens.size(0) // t
        
        # Transformer
        features = self.base(fusion_tokens, cam_label=cam_label) # [B*t, 129, 768]
        
        #-------------------Global Branch-------------
        b1_feat = self.b1(features)  # [B*t, 129, 768]
        global_feat = b1_feat[:, 0]
        
        global_feat = global_feat.unsqueeze(dim=2).unsqueeze(dim=3)
        a = F.relu(self.attention_conv(global_feat))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a_vals = a 
        
        a = F.softmax(a, dim=1)
        x = global_feat.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        
        global_feat = att_x.view(b, self.in_planes)
        feat = self.bottleneck(global_feat)
        
        #-------------------Local Branch-------------
        self.in_planes = 768  # 특징 차원
        self.parts_num = 6    # 부위 개수

        # 1. Features 처리
        features = features.view(b, t, 129, 768)  # [B, T, 129, 768]
        patch_tokens = features[:, :, 1:, :]      # [B, T, 128, 768], 클래스 토큰 제외
        patch_tokens = patch_tokens.permute(0, 2, 1, 3).contiguous().view(b, 128, t * self.in_planes)  # [B, 128, 3072]
        patch_features = patch_tokens.permute(0, 2, 1).contiguous().view(b, t * self.in_planes, 16, 8)  # [B, 3072, 16, 8]

        # 2. Heatmap 처리
        b, t, c, h, w = heatmap.shape  # [B, T, 6, 256, 128]
        heatmap_2d = heatmap.view(b * t, c, h, w)  # [BT, 6, 256, 128]
        parts_masks = F.interpolate(heatmap_2d, size=(16, 8), mode='bilinear')  # [BT, 6, 16, 8]
        parts_masks = parts_masks.view(b, t, c, 16, 8).mean(dim=1)  # [B, 6, 16, 8]

        # 3. 가시성 처리
        visibility_scores = torch.amax(heatmap, dim=(3, 4)).mean(dim=1)  # [B, 6]
        visibility_mask = visibility_scores > 0.1  # [B, 6]

        # 4. 부위별 임베딩 생성
        parts_embeddings = torch.einsum('bchw,bphw->bpc', patch_features, parts_masks)  # [B, 6, 3072]
        parts_embeddings = parts_embeddings * visibility_mask.unsqueeze(-1)  # [B, 6, 3072]

        # 5. 부위별 특징 정제
        refined_parts = []
        for i in range(self.parts_num):
            part_i = parts_embeddings[:, i, :].unsqueeze(1)  # [B, 1, 3072]
            refined_part = self.part_transformers[i](part_i)  # [B, 1, 3072], Transformer 블록 적용
            refined_part = refined_part.squeeze(1)  # [B, 3072]
            refined_parts.append(refined_part)
        refined_parts = torch.stack(refined_parts, dim=1)  # [B, 6, 3072]
                
        # # Part-based head
        # pixels_cls_scores = self.pixel_classifier(patch_features)  # [B, parts_num(6), 16, 8]
        # parts_masks = F.softmax(pixels_cls_scores, dim=1)  # [B, 6, 16, 8]
        # parts_masks = parts_masks / (parts_masks.sum(dim=(2, 3), keepdim=True) + 1e-6)
        # parts_embeddings = torch.einsum('bchw, bphw -> bpc', patch_features, parts_masks)  # [B, 6, 3072]
        
        # # Transformer Block for each part
        # refined_parts = []
        # for i in range(self.parts_num):  # now 6 parts
        #     part_i = parts_embeddings[:, i, :]  # [B, 3072]
        #     part_token = self.part_tokens[i].unsqueeze(0).expand(b, -1, -1)  # [B, 1, 3072]
        #     part_seq = torch.cat([part_token, part_i.unsqueeze(1)], dim=1)  # [B, 2, 3072]
        #     part_seq = self.part_transformers[i](part_seq)  # [B, 2, 3072]
        #     refined_part = part_seq[:, 0, :]  # refined part embedding: [B, 3072]
        #     refined_parts.append(refined_part)
        # refined_parts = torch.stack(refined_parts, dim=1)  # [B, 6, 3072]
        
        head_f = self.bottleneck_head(refined_parts[:, 0, :])
        torso_f = self.bottleneck_torso(refined_parts[:, 1, :])
        left_arm_f = self.bottleneck_left_arm(refined_parts[:, 2, :])
        right_arm_f = self.bottleneck_right_arm(refined_parts[:, 3, :])
        left_leg_f = self.bottleneck_left_leg(refined_parts[:, 4, :])
        right_leg_f = self.bottleneck_right_leg(refined_parts[:, 5, :])
            
        if self.training:
            Global_ID = self.classifier(feat)
            
            head_ID = self.classifier_head(head_f)
            torso_ID = self.classifier_torso(torso_f)
            left_arm_ID = self.classifier_left_arm(left_arm_f)
            right_arm_ID = self.classifier_right_arm(right_arm_f)
            left_leg_ID = self.classifier_left_leg(left_leg_f)
            right_leg_ID = self.classifier_right_leg(right_leg_f)
            
            return [Global_ID, head_ID, torso_ID, left_arm_ID, right_arm_ID, left_leg_ID, right_leg_ID], \
                   [global_feat, head_f, torso_f, left_arm_f, right_arm_f, left_leg_f, right_leg_f], a_vals
        else:
            return torch.cat([feat, head_f/6, torso_f/6, left_arm_f/6, right_arm_f/6, left_leg_f/6, right_leg_f/6], dim=1)  # [B, 3072*(1+6)]
            
    def load_param(self, trained_path, load=False):
        print("load_param실행")
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
                continue  # classifier 관련 파라미터는 로드하지 않음

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
                print(f"🔍 [Before Resizing] {new_k}: {v.shape} -> Expected: {expected_shape}")
                if v.shape[0] > expected_shape[0]:
                    v = v[:expected_shape[0], :, :]
                elif v.shape[0] < expected_shape[0]:
                    new_v = torch.randn(expected_shape)
                    new_v[:v.shape[0], :, :] = v
                    v = new_v

                print(f"✅ [After Resizing] {new_k}: {v.shape}")
                new_param_dict[new_k] = v
                continue
            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                new_param_dict[new_k] = v

        model_dict.update(new_param_dict)
        self.load_state_dict(model_dict, strict=False)
        print("✅ Checkpoint loaded successfully.")

