import torch
import torch.nn as nn
from functools import partial
from Dataloader import dataloader
from kim_js.ReID.KeyTransReID.heatmap_loader_previous import heatmap_dataloader
from vit_ID import PatchEmbed_overlap, trunc_normal_, TransReID
from token_fusion import fusiontoken

def run_transformer_with_fusion():
    # ---------------------------
    # 1. 데이터셋 로드
    # ---------------------------
    dataset_name = "Mars"
    image_loader, num_query, num_classes, cam_num, view_num, q_val_set, g_val_set = dataloader(dataset_name)
    heatmap_loader, _, _, _ = heatmap_dataloader(dataset_name)
    
    # 첫 번째 배치 가져오기
    images, pids, camids, _ = next(iter(image_loader))
    heatmaps, _, _ = next(iter(heatmap_loader))
    
    print("Original images shape:", images.shape)    # 예: [16, 4, 3, 256, 128]
    print("Original heatmaps shape:", heatmaps.shape)  # 예: [16, 4, 17, 256, 128]
    
    B, T, C, H, W = images.shape
    BT = B * T  # 총 프레임 수
    
    # ---------------------------
    # 2. Patch 임베딩 수행
    # ---------------------------
    # 이미지 patch 임베딩: in_chans=3
    img_patch_embed = PatchEmbed_overlap(img_size=(H, W), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=768)
    # heatmap patch 임베딩: in_chans=17
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(H, W), patch_size=(16, 16), stride_size=16, in_chans=17, embed_dim=768)
    
    # [B, T, ...] → [B*T, ...]
    images_reshaped = images.view(BT, C, H, W)             # [B*T, 3, 256, 128]
    heatmaps_reshaped = heatmaps.view(BT, heatmaps.shape[2], H, W)  # [B*T, 17, 256, 128]
    
    # 각 프레임에 대해 patch 토큰 추출: [B*T, num_patches, 768]
    img_tokens = img_patch_embed(images_reshaped)
    heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)
    
    # camids는 원래 [B, T] → flatten → [B*T]
    cam_labels = torch.tensor(camids).view(-1)
    
    # ---------------------------
    # 3. Token Fusion 수행
    # ---------------------------
    num_patches = 128  # PatchEmbed_overlap에 따라 결정됨
    embed_dim = 768
    fusion_module = fusiontoken(num_patches=num_patches, embed_dim=embed_dim, cam_num=cam_num)
    # fusion_tokens의 shape는 [B*T, 1 + num_patches, embed_dim]
    fusion_tokens = fusion_module(img_tokens, heatmap_tokens, cam_labels)
    print("Fusion tokens shape:", fusion_tokens.shape)  # 예상: [64, 129, 768] if B=16, T=4
    
    # ---------------------------
    # 4. TransReID 모델에 fusion token 입력하여 Transformer 실행
    # ---------------------------
    # 여기서 TransReID의 forward_features 함수는 입력이 3D일 경우(이미 fusion token) 추가 처리를 건너뛰도록 되어 있음.
    model = TransReID(img_size=224, patch_size=16, stride_size=16, in_chans=3,
                      num_classes=num_classes, embed_dim=embed_dim, depth=12, num_heads=12,
                      mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                      camera=cam_num, drop_path_rate=0.1, norm_layer=nn.LayerNorm, cam_lambda=3.0)
    
    # fusion token을 입력으로 넣으면, forward_features 함수에서 x.dim()==3 이므로 fusion token이 그대로 반환됨.
    output_tokens = model.forward_features(fusion_tokens, cam_labels)
    # 이후 Transformer 블록(예: norm) 적용
    final_output = model.norm(output_tokens)
    print("Transformer output shape:", final_output.shape)  # 예상: [B*T, 129, 768]
    print(final_output[0])
    
if __name__ == "__main__":
    run_transformer_with_fusion()
