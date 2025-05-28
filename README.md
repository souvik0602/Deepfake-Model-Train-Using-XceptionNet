# 🧠 Deepfake Detection using Xception and TensorFlow

This repository contains code for building and training a deep learning model to detect deepfake images using the Xception architecture. The model is trained using TensorFlow on a dataset of real and fake images.

---
## 📁 Dataset Structure
The dataset should be organized in the following structure:

- deepfake_dataset/
- - ├── train/
- - -│ ├── Real/
- - -│ └── Fake/
- -└── val/
- - -├── Real/
- - -└── Fake/

Each folder (`Real` and `Fake`) should contain images representing that class.

---

## 🚀 Model Architecture
We use the **Xception** architecture pre-trained on ImageNet, and add the following custom layers:

- `GlobalAveragePooling2D`
- `Dense(128, activation='relu')`
- `Dropout(0.5)`
- `Dense(1, activation='sigmoid')`

This model performs **binary classification**:  
- `0` → Real  
- `1` → Fake

---

## 🧪 Requirements
Install the dependencies with:
pip install -r requirements.txt

Required packages include:
- TensorFlow 2.x
- numpy
- opencv-python
- scikit-learn
- tqdm

🖥️ Hardware Used
- CPU: Intel® Xeon® Gold 6226R (64 cores)
- RAM: 384GB
- GPU: NVIDIA RTX A5000

📊 Sample Results
- Total Data:140000 (70000 Real +70000 Fake)
- Data Split:80:20
Metric	Value
- Training Accuracy	~71%
- Validation Accuracy	~81%
- Final Val Loss	~0.47
