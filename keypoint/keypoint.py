import os
import json
import openpifpaf
import numpy as np
import cv2
from PIL import Image

# 설정
CONFIDENCE_THRESHOLD = 0.1  # 신뢰도 기준 완화
NUM_KEYPOINTS = 17  # COCO keypoints (OpenPifPaf 기본 포맷)

# numpy JSON 직렬화 지원
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def preprocess_image(image_path):
    """이미지 전처리: CLAHE 적용 후 원본 해상도 유지"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        # CLAHE 적용 (대비 향상)
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    except Exception as e:
        print(f"❌ 이미지 로드 실패: {image_path} - {e}")
        return None
    return image_np


def extract_keypoints_from_image(img_path, predictor):
    """Keypoint 추출 (원본 이미지 좌표로 변환)"""
    image = preprocess_image(img_path)
    if image is None:
        # 이미지 로드 실패 시 keypoints를 (0,0,0)으로 채움
        return [[0, 0, 0] for _ in range(NUM_KEYPOINTS)], 0.0

    print(f"🖼️ Processing image: {img_path}")

    best_keypoints, best_conf = None, 0.0
    for scale in [1.0, 1.2, 1.5]:
        resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        try:
            predictions, _, _ = predictor.numpy_image(resized_image)
            if predictions:
                keypoints = predictions[0].data.copy()  # 복사해서 수정
                # 예측된 좌표는 resized_image 좌표 -> 원본 이미지 좌표로 보정
                keypoints[:, :2] = keypoints[:, :2] / scale

                confidences = keypoints[:, 2]
                avg_conf = confidences.mean() if confidences.size > 0 else 0.0

                if avg_conf > best_conf:
                    best_keypoints, best_conf = keypoints, avg_conf
        except Exception as e:
            print(f"❌ Prediction failed at scale {scale}: {e}")

    if best_keypoints is None or len(best_keypoints) == 0:
        print(f"⚠️ No predictions for {img_path}")
        return [[0, 0, 0] for _ in range(NUM_KEYPOINTS)], 0.0

    # COCO keypoints는 17개를 요구
    keypoints_list = best_keypoints.tolist()
    if len(keypoints_list) < NUM_KEYPOINTS:
        missing_count = NUM_KEYPOINTS - len(keypoints_list)
        keypoints_list += [[0, 0, 0]] * missing_count

    # CONFIDENCE_THRESHOLD를 넘는 keypoint만 고려하여 평균 신뢰도 계산
    conf_list = [kp[2] for kp in keypoints_list if kp[2] > CONFIDENCE_THRESHOLD]
    avg_conf = np.mean(conf_list) if len(conf_list) > 0 else 0.0
    return keypoints_list[:NUM_KEYPOINTS], avg_conf

def process_mars_dataset(root_dir, output_dir):
    predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')

    for phase in ['bbox_train', 'bbox_test']:
        phase_path = os.path.join(root_dir, phase)
        if not os.path.exists(phase_path):
            continue

        # **순서 보장**: 폴더명 순차 정렬
        for person_id in sorted(os.listdir(phase_path)):
            person_path = os.path.join(phase_path, person_id)
            if not os.path.isdir(person_path):
                continue

            keypoints_data = {}
            print(f"🚶 Processing person: {phase}/{person_id}")

            # **이미지 파일도 순서대로 처리**
            for img_file in sorted(os.listdir(person_path)):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(person_path, img_file)
                    keypoints, avg_conf = extract_keypoints_from_image(img_path, predictor)

                    # Keypoint 결과 저장
                    keypoints_data[img_file] = {
                        "keypoints": keypoints,
                        "avg_confidence": float(avg_conf)
                    }

            # 결과 저장
            save_dir = os.path.join(output_dir, phase, person_id)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "keypoints.json")
            with open(save_path, 'w') as f:
                json.dump(keypoints_data, f, indent=4, cls=NumpyEncoder)

            print(f"✅ Keypoints saved: {save_path}")


if __name__ == '__main__':
    dataset_root = '/home/user/kim_js/ReID/dataset/MARS'
    output_root = '/home/user/kim_js/ReID/dataset/MARS/keypoint'
    process_mars_dataset(dataset_root, output_root)
