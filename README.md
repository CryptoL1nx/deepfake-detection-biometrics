# Deepfake Detection using Biometric Face Recognition
### CS 599 Biometrics — Boston University Metropolitan College
**Author:** Caroline Bourdet | **GitHub:** [@CryptoL1nx](https://github.com/CryptoL1nx)

---

## Overview

This project implements a deepfake detection system based on biometric face 
recognition, reproducing and extending the findings of Ramachandran et al. (2021).

Instead of asking *"does this face look fake?"*, we ask *"does this face match 
who it claims to be?"*

A pretrained **ArcFace** model (ResNet-50 backbone) extracts a 512-dimensional 
facial embedding from each image. Real images of the same person produce similar 
embeddings. Deepfakes, despite looking visually convincing, corrupt the deep 
facial features enough to produce a detectable mismatch.

**Key results:**
| Experiment | Dataset | Result |
|---|---|---|
| Small-scale demo | Manual (Obama reference) | 6/6 correct (100%) |
| Face verification | LFW (135 pairs) | AUC = 0.918 |
| Deepfake detection | FaceForensics++ FaceSwap | 50/50 correct (100%) |

---

## How It Works

Reference image (real) ──► ArcFace ──► 512-dim embedding ──┐
├──► Cosine similarity ──► Decision
Test image (real/fake) ──► ArcFace ──► 512-dim embedding ──┘
Score ≥ 0.60 → GENUINE ✅
Score < 0.60 → FAKE / IMPOSTOR ❌
---

## Project Structure
deepfake-detection-biometrics/
│
├── deepfake_detection.ipynb      # Main notebook — all experiments
├── extract_frames.py             # Extract frames from FF++ videos
├── faceforensics_download.py     # Download FaceForensics++ dataset
│
├── score_distribution.png        # Score distribution plot
├── roc_curve_lfw.png             # ROC curve on LFW dataset
├── combined_score_distribution.png # Combined genuine/impostor/deepfake
├── ff_detection_results.png      # FaceForensics++ visual results
├── results_analysis.png          # Comprehensive results analysis
│
├── .gitignore
└── LICENSE
---

## Requirements

- Python 3.11
- Windows 10/11 (tested), macOS, Linux

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/CryptoL1nx/deepfake-detection-biometrics.git
cd deepfake-detection-biometrics
```

### 2. Create a virtual environment with Python 3.11
```bash
py -3.11 -m venv deepfake_env
.\deepfake_env\Scripts\activate    # Windows
source deepfake_env/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install deepface tf-keras opencv-python matplotlib scikit-learn requests jupyter notebook tqdm
```

### 4. Launch the notebook
```bash
jupyter notebook
```
Open `deepfake_detection.ipynb` and run cells top to bottom.

---

## Datasets

### Sample images (automatic)
The notebook automatically uses manually collected images of public figures
for the initial small-scale demonstration. No download needed.

### LFW Dataset (for ROC curve evaluation)
Download from Kaggle:
'''
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
'''

Extract and place the following 5 folders into `lfw/`:
- `George_W_Bush/`
- `Colin_Powell/`
- `Tony_Blair/`
- `Donald_Rumsfeld/`
- `Gerhard_Schroeder/`

### FaceForensics++ (for real deepfake detection)
Request access and download using the provided script:
```bash
# Download deepfake videos (FaceSwap, compressed)
python faceforensics_download.py C:\cs599\deepfake\faceforensics --server EU2 -d FaceSwap -c c40 -t videos -n 10

# Download original real videos
python faceforensics_download.py C:\cs599\deepfake\faceforensics --server EU2 -d original -c c40 -t videos -n 10

# Extract frames
python extract_frames.py
```

Access request form: `https://github.com/ondyari/FaceForensics`

---

## Results

### Score Distribution
![Score Distribution](score_distribution.png)

### ROC Curve (LFW)
![ROC Curve](roc_curve_lfw.png)

### Combined Distribution (Genuine vs Impostor vs Deepfake)
![Combined Distribution](combined_score_distribution.png)

### FaceForensics++ Detection Results
![FF++ Results](ff_detection_results.png)

### Comprehensive Analysis
![Results Analysis](results_analysis.png)

---

## Key Findings

1. **100% detection accuracy** on FaceForensics++ FaceSwap deepfakes
2. **AUC = 0.918** on LFW face verification (paper reports 0.98 on Celeb-DF)
3. **FPR = 0.000** at threshold 0.60 — zero false positives
4. **No deepfake training data required** — the system only needs genuine 
   reference images
5. Deepfakes score significantly lower than genuine pairs 
   (mean 0.28 vs 0.52), confirming the paper's core finding

---

## References

- Ramachandran, S., Nadimpalli, A. V., & Rattani, A. (2021). *An Experimental 
  Evaluation on Deepfake Detection using Deep Face Recognition.* ICCST 2021.

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular 
  Margin Loss for Deep Face Recognition.* CVPR 2019.

- Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. 
  (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019.

- Huang, G., Mattar, M., Berg, T., & Learned-Miller, E. (2007). *Labeled Faces 
  in the Wild.* ECCV Workshop 2007.

---

## Dataset Licenses

- **FaceForensics++**: Used under the 
  [FaceForensics Terms of Use](http://kaldir.vc.in.tum.de/faceforensics_tos.pdf)
- **LFW**: Used for research and educational purposes only
- **This code**: MIT License — see [LICENSE](LICENSE)

---

## Course Information

**Course:** CS 599 Biometrics — Boston University Metropolitan College  
**Instructor:** Prof. Zoran Djordjevic  
**Semester:** Spring 2026  
**Assignment:** Final Project — Assignment 13 (200 pts)