# KeyRe-ID

**KeyRe-ID** is a dual-branch video-based person Re-Identification (Re-ID) framework that leverages human keypoints and transformer architectures to enhance spatio-temporal feature representation. This project improves robustness to pose variation, occlusions, and background clutter in Video-based Re-ID tasks.

## 📖 Project Overview

KeyRe-ID consists of two complementary branches:

- **Global Branch:**  
  Extracts clip-level global features by applying frame-level attention based on the [CLS] token. This branch models temporal dependencies across frames to generate robust global representations.

- **Local Branch:**  
  Focuses on fine-grained, part-level features using keypoint-guided part segmentation (KPSM). Human body parts are dynamically segmented via pose estimation, and part-specific transformers capture local temporal coherence.

KeyRe-ID integrates:

- Keypoint-guided part-aware feature extraction  
- Transformer-based temporal modeling  
- Temporal Clip Shift and Shuffle (TCSS) for temporal robustness  
- Attention mechanisms for frame-level importance learning  
- Keypoint-guided Part Segmentation Module (KPSM) for dynamic, semantically meaningful body part extraction based on pose estimation  

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/JinSeong0115/KeyRe-ID.git
cd KeyRe-ID

```

## 🚀 Usage

### Training and Evaluation
To train or evaluate the model on supported Video-based Re-ID benchmarks (e.g., MARS, iLIDS-VID):
```bash
python Key_ReID.py --Dataset_name 'Dataset_name' --ViT_path 'pretrained_model.pth'
```
Example for MARS dataset:
```bash
python Key_ReID.py --Dataset_name 'Mars' --ViT_path 'path_to_pretrained_model.pth'
```

## ✨ Key Features Summary

✔️ Dual-branch structure for complementary Global and Local semantic feature learning  
✔️ Keypoint-guided dynamic part segmentation using KPSM for anatomically meaningful body part extraction  
✔️ Transformer-based temporal modeling with frame-level attention for capturing long-range temporal dependencies  
✔️ Temporal Clip Shift and Shuffle (TCSS) module to mitigate temporal misalignment and enhance robustness  
✔️ Optimized for video-based person Re-ID benchmarks under occlusion, pose variation, and illumination changes  


## 🙏 Acknowledgement
- Thanks to AishahAADU, using some implementation from [AishahAADU's repository](https://github.com/AishahAADU/VID-Trans-ReID)  


