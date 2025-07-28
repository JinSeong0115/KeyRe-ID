# KeyRe-ID

**KeyRe-ID** is a dual-branch video-based person Re-Identification (Re-ID) framework that leverages human keypoints and Vision Transformer (ViT) architectures to improve identity recognition across time and space. This project is designed to be robust against pose variation, occlusion, and cross-view misalignment in multi-camera environments.

---

## Overview

KeyRe-ID tackles the challenge of video-based person re-identification by combining global semantic features and local body-part-aware cues. It uses a Vision Transformer (ViT) to encode spatio-temporal features and integrates pose keypoints to guide fine-grained part-based representations.

This design allows the model to:

- Capture clip-level identity semantics
- Adapt to pose and motion variations
- Improve matching under occlusion and misalignment

KeyRe-ID achieves state-of-the-art performance on MARS and iLIDS-VID datasets, demonstrating its effectiveness in real-world Re-ID scenarios.

---

## Architecture

The overall architecture of KeyRe-ID includes the following components:

- **ViT Backbone**  
  Encodes each sampled video clip into patch tokens and [CLS] tokens across frames.

- **Global Branch**  
  Applies Transformer-based temporal attention to [CLS] tokens, producing a clip-level identity embedding.

- **Local Branch**  
  Utilizes pose keypoints (via a pretrained pose estimator) to create semantic heatmaps. These heatmaps guide patch-level aggregation into part-aware features through the **Keypoint-guided Part Segmentation (KPS)** module.

- **Temporal Clip Shift & Shuffle (TCSS)**  
  Perturbs the order of patch tokens across frames to enforce temporal invariance and prevent overfitting to fixed motion patterns.

![KeyRe-ID Framework](assets/keyreid_framework.png)

---

## Training and Inference

KeyRe-ID is trained with a composite objective that combines:

- Cross-entropy loss (with label smoothing)
- Triplet loss
- Center loss
- Attention regularization loss

During inference, global and part-level embeddings are concatenated to generate the final descriptor, which is used for retrieval via Euclidean distance.

---

## Performance

| Dataset    | mAP (%) | Rank-1 (%) | Rank-5 (%) |
|------------|---------|------------|------------|
| MARS       | 91.73   | 97.32      | –          |
| iLIDS-VID  | –       | 96.00      | 100.00     |

---






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


