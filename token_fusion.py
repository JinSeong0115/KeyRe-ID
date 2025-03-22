import torch
import torch.nn as nn
from heatmap_loader import heatmap_dataloader
from vit_ID import PatchEmbed_overlap

class FusionToken(nn.Module):
    def __init__(self, num_patches, embed_dim, cam_num):
        """
        Args:
            num_patches: 각 이미지에서 추출된 패치 개수 (예: 128)
            embed_dim: patch 임베딩 차원 (예: 768)
            cam_num: 카메라 종류 수 (카메라 임베딩 용)
        """
        super(FusionToken, self).__init__()
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.cam = nn.Parameter(torch.zeros(cam_num, 1, embed_dim))
        
        self.cam_lambda = 3.0
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cam, std=0.02)
    
    def forward(self, img_tokens, heatmap_tokens, cam_labels):
        """
        Args:
            img_tokens: [B*T, num_patches, embed_dim] - 이미지 patch 임베딩
            heatmap_tokens: [B*T, num_patches, embed_dim] - heatmap patch 임베딩
            cam_labels: [B*T] - 각 프레임의 카메라 id
        Returns:
            tokens: [B*T, 1 + num_patches, embed_dim] - fusion된 토큰 시퀀스
        """
        B_T, num_patches, embed_dim = img_tokens.shape

        # 1. 이미지와 heatmap 토큰의 element-wise 덧셈
        fused_tokens = img_tokens + heatmap_tokens

        # 2. CLS 토큰 구성: CLS 토큰 + 전용 positional embedding + cam embedding
        cls_token = self.cls_token.expand(B_T, -1, -1)
        pos_cls = self.pos_embed[:, 0, :].unsqueeze(0).expand(B_T, 1, embed_dim)
        cam_embed = self.cam[cam_labels]  # [B*T, 1, embed_dim]
        cls_fused = cls_token + pos_cls + self.cam_lambda*cam_embed

        # 3. 각 패치 토큰 구성: fused patch + positional embedding + cam embedding
        pos_patch = self.pos_embed[:, 1:, :].expand(B_T, num_patches, embed_dim)
        cam_embed = cam_embed.expand(-1, num_patches, -1)
        patch_fused = fused_tokens + pos_patch + self.cam_lambda*cam_embed

        # 4. CLS 토큰과 patch 토큰들을 concat
        tokens = torch.cat([cls_fused, patch_fused], dim=1)
        return tokens

def test_token_fusion_on_real_data():
    dataset_name = "Mars"
    # heatmap_loader.py를 사용하여 이미지와 heatmap 데이터를 함께 로드
    train_loader, num_query, num_classes, cam_num, num_train, q_val_set, g_val_set = heatmap_dataloader(dataset_name)
    
    # CombinedDataset_inderase의 __getitem__가 반환하는 값은 다음과 같습니다:
    # (imgs, heatmaps, pid, target_cam, labels, selected_img_paths)
    images, heatmaps, pids, camids, labels, img_paths = next(iter(train_loader))
    
    print("Original images shape:", images.shape)   # 예: [B, seq_len, 3, 256, 128]
    print("Original heatmaps shape:", heatmaps.shape) # 예: [B, seq_len, 6, 256, 128]
    
    B, T, C, H, W = images.shape
    BT = B * T

    # 패치 임베딩 설정
    num_patches = 128
    embed_dim = 768

    # 이미지와 heatmap에 대해 PatchEmbed_overlap 모듈 생성
    img_patch_embed = PatchEmbed_overlap(img_size=(H, W), patch_size=(16, 16), stride_size=16, in_chans=3, embed_dim=embed_dim)
    # 여기서 heatmap의 채널 수는 6으로 설정합니다.
    heatmap_patch_embed = PatchEmbed_overlap(img_size=(H, W), patch_size=(16, 16), stride_size=16, in_chans=6, embed_dim=embed_dim)
    
    # [B, seq_len, ...] 형태를 [B*T, ...]로 reshape
    images_reshaped = images.view(BT, C, H, W)
    # heatmaps_reshaped: (B*T, 6, H, W)
    heatmaps_reshaped = heatmaps.view(BT, heatmaps.shape[2], H, W)
    
    # 각 프레임에 대해 patch 토큰 추출
    img_tokens = img_patch_embed(images_reshaped)
    heatmap_tokens = heatmap_patch_embed(heatmaps_reshaped)
    
    # cam 정보 flatten: camids는 리스트의 리스트 형태이므로 flatten합니다.
    cam_labels_list = []
    for cam_list in camids:
        cam_labels_list.extend(cam_list)
    cam_labels = torch.tensor(cam_labels_list)  # [B*T]

    # Fusion token 생성 및 결합
    fusion_module = FusionToken(num_patches=num_patches, embed_dim=embed_dim, cam_num=cam_num)
    fusion_tokens = fusion_module(img_tokens, heatmap_tokens, cam_labels)
    
    print("Fusion tokens shape:", fusion_tokens.shape)
    print(fusion_tokens[0])

if __name__ == "__main__":
    test_token_fusion_on_real_data()
