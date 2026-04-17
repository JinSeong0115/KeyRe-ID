import os
import json
import openpifpaf
import numpy as np
from PIL import Image
import argparse

def extract_keypoints_from_image(image_path, predictor, keypoint_threshold=0.005):
    """
    - Detect keypoints in the original image without upscaling.
    - Detect more keypoints by lowering the keypoint confidence threshold.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Run a prediction
    predictions, _, _ = predictor.numpy_image(image_np)

    # Extract keypoints
    if predictions:
        best_pred = max(predictions, key=lambda p: p.score)
        keypoints = [
            [float(x), float(y), float(conf)] if conf > keypoint_threshold else [0.0, 0.0, 0.0]
            for x, y, conf in best_pred.data
        ]
    else:
        # Create a list of 17 zero-keypoints if no person is detected
        keypoints = [[0.0, 0.0, 0.0]] * 17
    
    return keypoints

def process_dataset(dataset_path, output_dir, max_images=None, skip_existing=False, log_every=100):
    """
    - Extracting and saving keypoints from all folders inside dataset_path.
    - Save keypoint data in JSON format with a .pose extension.
    """
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
    processed = 0

    for root, dirs, files in os.walk(dataset_path):
        dirs[:] = sorted(d for d in dirs if d not in {'keypoints', 'heatmap', 'heatmaps'})
        if not files:
            continue

        relative_path = os.path.relpath(root, dataset_path)
        output_folder = os.path.join(output_dir, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        for img_file in sorted(files):
            if img_file.lower().endswith(('.jpg', '.png')):
                if max_images is not None and processed >= max_images:
                    print(f"Reached max_images={max_images}.")
                    return
                img_path = os.path.join(root, img_file)
                pose_filename = f"{os.path.splitext(img_file)[0]}.pose"
                pose_output_path = os.path.join(output_folder, pose_filename)
                if skip_existing and os.path.exists(pose_output_path):
                    processed += 1
                    continue
                try:
                    keypoints = extract_keypoints_from_image(img_path, predictor)

                    if not keypoints or all(kp == [0.0, 0.0, 0.0] for kp in keypoints):
                        print(f"Warning: No keypoints detected in {img_path}")

                    # --- MODIFIED PART ---
                    # Changed the output filename to end with .pose and removed the '_keypoints' suffix
                    # for compatibility with the next script.
                    with open(pose_output_path, 'w') as f:
                        # The content is still saved in JSON format.
                        json.dump(keypoints, f, indent=4)

                    processed += 1
                    if log_every and processed % log_every == 0:
                        print(f"Processed {processed} images. Latest: {pose_output_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

    print(f"Processed {processed} images in total.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract keypoints from images using OpenPifPaf.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the root directory of the image dataset.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where keypoint files will be saved.')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Optional limit for smoke tests.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip images whose .pose output already exists.')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Progress logging interval. Use 0 to disable periodic logs.')
    args = parser.parse_args()

    process_dataset(
        args.dataset_path,
        args.output_dir,
        max_images=args.max_images,
        skip_existing=args.skip_existing,
        log_every=args.log_every,
    )
    print("Keypoint extraction complete.")


# python ./keypoint/extract_keypoint.py --dataset_path ./data/MARS --output_dir ./data/MARS/keypoints
