# import os
# import cv2
# import torch
# import numpy as np
# import argparse
# from tqdm import tqdm
# from torch.utils.data import DataLoader

# # 사용자 파일 import
# from heatmap_loader import heatmap_dataloader
# from KeyRe_ID_model_part import KeyRe_ID

# # ───── 설정 ─────
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# def denormalize(img_tensor):
#     """Normalize된 텐서를 원본 이미지(BGR)로 복원"""
#     img = img_tensor.cpu().numpy().transpose(1, 2, 0)
#     img = img * IMAGENET_STD + IMAGENET_MEAN
#     img = np.clip(img, 0, 1) * 255
#     return img.astype(np.uint8)[..., ::-1]

# def overlay_attention(img_bgr, attention_map):
#     """Attention Map을 부드럽게 뭉개서(Smooth) 이미지 위에 그리기"""
#     H, W = img_bgr.shape[:2]
    
#     # 1. 확대 (Cubic Interpolation으로 1차 부드러움)
#     heatmap = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)
    
#     # 2. 정규화
#     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
#     heatmap = np.uint8(255 * heatmap)
    
#     # 3. Gaussian Blur로 뭉개기 (커널 사이즈 21x21) -> 구름처럼 보이게 함
#     heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
#     # 4. 컬러맵 적용 (JET)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
#     # 5. 합치기
#     overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
#     return overlay

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_name", default="MARS", type=str)
#     parser.add_argument("--dataset_root", default="./data", type=str)
#     parser.add_argument("--ViT_path", default="./weights/jx_vit_base_p16_224-80ecf9dd.pth", type=str)
#     parser.add_argument("--trained_weight", default="./weights/MARSbest_mAP.pth", type=str)
#     parser.add_argument("--save_dir", default="./visualization_results", type=str)
#     args = parser.parse_args()

#     os.makedirs(args.save_dir, exist_ok=True)

#     # 1. 데이터 로드 (순서 중요!)
#     print("Loading Data...")
#     _, _, num_classes, camera_num, _, query_loader, _ = heatmap_dataloader(args.dataset_name, args.dataset_root)
    
#     # 2. 모델 로드
#     print(f"Loading model weights from {args.trained_weight}...")
#     model = KeyRe_ID(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.ViT_path)
    
#     if os.path.exists(args.trained_weight):
#         model.load_state_dict(torch.load(args.trained_weight), strict=False)
#     else:
#         print(f"Error: Trained weight not found at {args.trained_weight}")
#         return
        
#     model = model.cuda().eval()

#     # 3. 시각화 루프
#     print("Start Visualization...")
#     part_names = ["Head", "Torso", "L-Arm", "R-Arm", "L-Leg", "R-Leg"]

#     for i, (imgs, heatmaps, pid, camids, _) in enumerate(tqdm(query_loader)):
        
#         # [옵션] 너무 많으면 중간에 멈추기 (전체 다 하려면 주석 처리)
#         # if i >= 50: break 
        
#         # 차원 보정
#         if len(imgs.shape) == 4:
#             imgs = imgs.unsqueeze(0)
#         if len(heatmaps.shape) == 4:
#             heatmaps = heatmaps.unsqueeze(0)
            
#         imgs = imgs.cuda()
#         heatmaps = heatmaps.cuda()
        
#         if isinstance(camids, torch.Tensor):
#             camids = camids.cuda()
        
#         with torch.no_grad():
#             output = model(imgs, heatmaps, pid, cam_label=camids)
            
#             # 모델 수정 여부 체크
#             if isinstance(output, tuple):
#                 heatmap_weights = output[1]
#             else:
#                 print("\n[Error] Model returns a single value, not a tuple.")
#                 print("Please modify 'KeyRe_ID_model.py' to return (features, weights).")
#                 return

#         # 첫 번째 배치의 데이터 가져오기
#         # 모델에서 이미 permute를 했거나 안 했거나에 따라 처리
#         if heatmap_weights.shape[-1] == 6: 
#              heatmap_weights = heatmap_weights.permute(0, 2, 1)

#         weights = heatmap_weights[0] 
#         orig_img_tensor = imgs[0, 0] 
#         img_bgr = denormalize(orig_img_tensor)
#         vis_list = [img_bgr]
        
#         for p in range(6):
#             try:
#                 att_map = weights[p].view(16, 8).cpu().numpy()
#             except RuntimeError:
#                 continue
            
#             vis_part = overlay_attention(img_bgr, att_map)
#             cv2.putText(vis_part, part_names[p], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#             vis_list.append(vis_part)
            
#         final_vis = np.hstack(vis_list)
        
#         # PID/CamID 값 추출
#         if isinstance(pid, torch.Tensor): pid_val = int(pid.item())
#         elif isinstance(pid, (list, tuple)): pid_val = int(pid[0])
#         else: pid_val = int(pid)
            
#         if isinstance(camids, torch.Tensor): cam_val = int(camids.item())
#         elif isinstance(camids, (list, tuple)): cam_val = int(camids[0])
#         else: cam_val = int(camids)

#         save_path = os.path.join(args.save_dir, f"ID{pid_val:04d}_cam{cam_val}_vis.jpg")
#         cv2.imwrite(save_path, final_vis)
        
#     print(f"All Visualizations Saved to {args.save_dir}")

# if __name__ == "__main__":
#     main()

import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 사용자 파일 import (사용자님이 올려주신 코드 기준)
from heatmap_loader import heatmap_dataloader
from KeyRe_ID_model_part import KeyRe_ID

# ───── 설정 ─────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# [수정 1] 시각화하고 싶은 대상 리스트 (ID, CamID)
TARGET_SAMPLES = [
    (48, 4),  # ID 100, Camera 4
]

def denormalize(img_tensor):
    """Normalize된 텐서를 원본 이미지(BGR)로 복원"""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)[..., ::-1]

def overlay_attention(img_bgr, attention_map):
    """Attention Map을 부드럽게 뭉개서(Smooth) 이미지 위에 그리기"""
    H, W = img_bgr.shape[:2]
    
    # 1. 확대 (Cubic Interpolation으로 1차 부드러움)
    heatmap = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # 2. 정규화
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    
    # 3. Gaussian Blur로 뭉개기
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # 4. 컬러맵 적용 (JET)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 5. 합치기
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    return overlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="MARS", type=str)
    parser.add_argument("--dataset_root", default="./data", type=str)
    parser.add_argument("--ViT_path", default="./weights/jx_vit_base_p16_224-80ecf9dd.pth", type=str)
    parser.add_argument("--trained_weight", default="./weights/MARSbest_mAP.pth", type=str)
    parser.add_argument("--save_dir", default="./visualization_results", type=str)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 데이터 로드
    print("Loading Data...")
    _, _, num_classes, camera_num, _, query_loader, _ = heatmap_dataloader(args.dataset_name, args.dataset_root)
    
    # 2. 모델 로드
    print(f"Loading model weights from {args.trained_weight}...")
    model = KeyRe_ID(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.ViT_path)
    
    if os.path.exists(args.trained_weight):
        model.load_state_dict(torch.load(args.trained_weight), strict=False)
    else:
        print(f"Error: Trained weight not found at {args.trained_weight}")
        return
        
    model = model.cuda().eval()

    # 3. 시각화 루프
    print(f"Start Visualization for targets: {TARGET_SAMPLES}")
    
    # 진행률 표시
    for i, (imgs, heatmaps, pid, camids, _) in enumerate(tqdm(query_loader)):
        
        # ---------------------------------------------------------
        # [수정 2] ID와 CamID를 먼저 추출해서 타겟인지 확인
        # ---------------------------------------------------------
        if isinstance(pid, torch.Tensor): pid_val = int(pid.item())
        elif isinstance(pid, (list, tuple)): pid_val = int(pid[0])
        else: pid_val = int(pid)
            
        if isinstance(camids, torch.Tensor): cam_val = int(camids.item())
        elif isinstance(camids, (list, tuple)): cam_val = int(camids[0])
        else: cam_val = int(camids)

        # 타겟 리스트에 없으면 건너뛰기 (속도 최적화)
        if (pid_val, cam_val) not in TARGET_SAMPLES:
            continue
            
        # ---------------------------------------------------------
        # 타겟을 찾았으므로 처리 시작
        # ---------------------------------------------------------
        
        # 차원 보정
        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(0)
        if len(heatmaps.shape) == 4:
            heatmaps = heatmaps.unsqueeze(0)
            
        imgs = imgs.cuda()
        heatmaps = heatmaps.cuda()
        
        if isinstance(camids, torch.Tensor):
            camids = camids.cuda()
        
        with torch.no_grad():
            output = model(imgs, heatmaps, pid, cam_label=camids)
            
            if isinstance(output, tuple):
                heatmap_weights = output[1]
            else:
                print("\n[Error] Model returns a single value.")
                return

        if heatmap_weights.shape[-1] == 6: 
             heatmap_weights = heatmap_weights.permute(0, 2, 1)

        weights = heatmap_weights[0] 
        orig_img_tensor = imgs[0, 0] 
        img_bgr = denormalize(orig_img_tensor)
        vis_list = [img_bgr]
        
        for p in range(6):
            try:
                att_map = weights[p].view(16, 8).cpu().numpy()
            except RuntimeError:
                continue
            
            vis_part = overlay_attention(img_bgr, att_map)
            
            # [수정 3] 글자 쓰는 부분 주석 처리 (텍스트 제거)
            # cv2.putText(vis_part, part_names[p], ...) 
            
            vis_list.append(vis_part)
            
        final_vis = np.hstack(vis_list)
        
        # 파일 저장 (인덱스 i를 추가하여 중복 방지)
        save_path = os.path.join(args.save_dir, f"ID{pid_val:04d}_cam{cam_val}_idx{i}_vis.jpg")
        cv2.imwrite(save_path, final_vis)
        
    print(f"Target Visualizations Saved to {args.save_dir}")

if __name__ == "__main__":
    main()
