import torch
from vit_ID import PatchEmbed_overlap
from heatmap_loader import heatmap_dataloader

def process_image_batch(patch_embed, images):
    """
    이미지 배치에 대해 patch embedding을 수행합니다.
    입력 images shape: [B, seq_len, 3, H, W]
    → 각 프레임마다 patch embedding을 적용하여
       출력 shape: [B * seq_len, num_patches, embed_dim]
    """
    B, T, C, H, W = images.shape
    images_reshaped = images.view(B * T, C, H, W)
    patches = patch_embed(images_reshaped)
    return patches

def process_heatmap_batch(patch_embed, heatmaps):
    """
    heatmap 배치에 대해 patch embedding을 수행합니다.
    입력 heatmaps shape: [B, seq_len, 17, H, W]
    → 각 프레임마다 patch embedding을 적용하여
       출력 shape: [B * seq_len, num_patches, embed_dim]
    """
    B, T, C, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B * T, C, H, W)
    patches = patch_embed(heatmaps_reshaped)
    return patches

def test_token_fusion_on_real_data():
    # 데이터셋 이름 (예: Mars)
    dataset_name = "Mars"
    
    # heatmap_dataloader를 통해 CombinedDataset_inderase의 결과를 로드합니다.
    # 이 CombinedDataset_inderase의 collate 함수는 5개의 값을 반환합니다.
    from heatmap_loader import heatmap_dataloader
    loader = heatmap_dataloader(dataset_name)
    
    # 첫 번째 배치에서 반환값 5개를 unpack 합니다.
    imgs, heatmaps, pids, camids, heatmap_files = next(iter(loader))
    
    print("Original images shape:", imgs.shape)       # 예: [B, seq_len, 3, 256, 128]
    print("Original heatmaps shape:", heatmaps.shape)   # 예: [B, seq_len, 17, 256, 128]
    
    B, T, C, H, W = imgs.shape
    BT = B * T
    num_patches = 128   # 예시값 (PatchEmbed_overlap에 따라 달라짐)
    embed_dim = 768
    
    # PatchEmbed_overlap 모듈 생성
    from vit_ID import PatchEmbed_overlap
    img_patch_embed = PatchEmbed_overlap(img_size=(H, W), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=embed_dim)
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(H, W), patch_size=(16, 16), stride_size=16, in_chans=17, embed_dim=embed_dim)
    
    # [B, seq_len, ...] -> [B*T, ...] reshape
    imgs_reshaped = imgs.view(BT, C, H, W)
    heatmaps_reshaped = heatmaps.view(BT, heatmaps.shape[2], H, W)
    
    # 각 프레임별 patch embedding 수행
    img_tokens = img_patch_embed(imgs_reshaped)         # [B*T, num_patches, embed_dim]
    heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)  # [B*T, num_patches, embed_dim]
    
    # camids는 [B, seq_len] -> flatten -> [B*T]
    cam_labels = torch.tensor(camids).view(-1)
    
    # Fusion token 생성 모듈
    from token_fusion import fusiontoken
    fusion_module = fusiontoken(num_patches=num_patches, embed_dim=embed_dim, cam_num=len(torch.unique(cam_labels)))
    
    # Fusion 수행: 최종 토큰 시퀀스 [B*T, 1+num_patches, embed_dim]
    fusion_tokens = fusion_module(img_tokens, heatmap_tokens, cam_labels)
    
    print("Fusion tokens shape:", fusion_tokens.shape)
    print("Fusion token[0] sample:", fusion_tokens[0])
    
if __name__ == "__main__":
    test_token_fusion_on_real_data()

