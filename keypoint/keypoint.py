import os
import json
import openpifpaf
import numpy as np
from PIL import Image

def extract_keypoints_from_image(image_path, predictor, keypoint_threshold=0.005):
    """
    - 업스케일링 없이 원본 이미지에서 keypoint 검출.
    - keypoint confidence threshold를 더 낮춰 더 많은 keypoint 검출.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 예측 실행
    predictions, _, _ = predictor.numpy_image(image_np)

    # keypoints 추출
    if predictions:
        best_pred = max(predictions, key=lambda p: p.score)
        keypoints = [
            [float(x), float(y)] if conf > keypoint_threshold else [0.0, 0.0]
            for x, y, conf in best_pred.data
        ]
    else:
        keypoints = []
    
    return keypoints

def process_dataset(dataset_root, output_dir):
    """
    - dataset_root 내부의 모든 폴더에서 keypoint를 추출 및 저장.
    - 원본 폴더 구조를 그대로 반영하여 keypoint JSON 저장.
    """
    predictor = openpifpaf.Predictor(checkpoint='resnet50')

    for root, dirs, files in os.walk(dataset_root):
        if not files:
            continue

        relative_path = os.path.relpath(root, dataset_root)
        output_folder = os.path.join(output_dir, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(root, img_file)
                try:
                    keypoints = extract_keypoints_from_image(img_path, predictor)

                    # keypoints가 비어 있으면 경고 메시지 출력
                    if not keypoints or all(kp == [0.0, 0.0] for kp in keypoints):
                        print(f"⚠️ Warning: No keypoints detected in {img_path}")

                    json_filename = f"{os.path.splitext(img_file)[0]}_keypoints.json"
                    json_output_path = os.path.join(output_folder, json_filename)

                    with open(json_output_path, 'w') as f:
                        json.dump(keypoints, f, indent=4)
                    
                    print(f"✅ Saved keypoints for {img_file} at {json_output_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
