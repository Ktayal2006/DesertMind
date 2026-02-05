# 🌵 DesertMind  
### Off-Road Semantic Segmentation & Analytics System

DesertMind is a deep learning–based semantic segmentation system for off-road and desert environments.  
It combines a GPU-optimized training pipeline with a desktop analytics dashboard for real-time terrain analysis.

The project is designed for applications in autonomous navigation, robotics, and scene understanding in unstructured environments.

---

## 📌 Features

- DeepLabV3 semantic segmentation model
- MobileNetV3 Large backbone
- Mixed Precision (AMP) training
- Early stopping for efficient learning
- Custom dataset loader with label remapping
- Interactive PyQt6 dashboard
- Real-time inference visualization
- Class distribution and confidence analysis
- Inference latency monitoring
- Cross-platform support (Windows / macOS / Linux)

---

## 🧠 Motivation

Off-road environments lack structured visual cues such as:

- Lane markings
- Clear boundaries
- Uniform textures

Instead, they contain:

- Sand, rocks, dirt, vegetation
- Irregular lighting
- Texture variations
- Unpredictable terrain

Traditional vision systems struggle in such conditions.

DesertMind learns pixel-level terrain semantics, enabling intelligent decision-making for autonomous and robotic systems operating in harsh environments.

---

## 🏗️ System Overview

```
Input Image
↓
Preprocessing
↓
MobileNetV3 Feature Encoder
↓
DeepLabV3 ASPP Module
↓
Segmentation Head
↓
Prediction Map
↓
Visualization & Analytics
```

---

## 📂 Repository Structure

```
desertmind/
│
├── app.py # GUI dashboard and inference system
├── dataset.py # Custom dataset loader and label mapping
├── train_deeplab.py # Training pipeline
├── requirements.txt # Project dependencies
└── README.md # Documentation
```

---

## 🧪 Model Architecture

| Component | Description |
|-----------|-------------|
| Architecture | DeepLabV3 |
| Backbone | MobileNetV3 Large |
| Loss Function | Cross Entropy |
| Optimizer | AdamW |
| Metrics | Mean IoU, Pixel Accuracy |
| Classes | 10 |

The MobileNetV3 backbone provides lightweight feature extraction, while DeepLabV3 enables multi-scale context learning using atrous convolutions.

---

## 📁 Dataset Format

The dataset directory must follow this structure:

```
Offroad_Segmentation_Training_Dataset/
├── train/
│ ├── Color_Images/
│ └── Segmentation/
├── val/
│ ├── Color_Images/
│ └── Segmentation/
└── test/ (optional)
```

- Images must be in PNG format.
- Each mask must have the same filename as its corresponding image.
- Raw labels are remapped internally to 10 training classes.

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies:

- Python ≥ 3.11
- PyTorch ≥ 2.0
- TorchVision ≥ 0.15
- PyQt6
- NumPy
- Pillow
- Matplotlib

---

## 🚀 Installation

1️⃣ Create Virtual Environment

```bash
py -3.11 -m venv venv
.\venv\Scripts\activate
```

2️⃣ Install Packages

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

**Step 1: Configure Dataset Path**

Open `train_deeplab.py` and edit:

```python
ROOT = r"path/to/Offroad_Segmentation_Training_Dataset"
```

**Step 2: Run Training**

```bash
python train_deeplab.py
```

**Training Output Example**

```
Using device: cuda
Train size: 2857 Val size: 317

Epoch 1/10 | train loss ... | val loss ... | val mIoU ... | pixel acc ...
Saved best model: 0.42
```

**Training Features**

- Automatic mixed precision
- Early stopping (patience-based)
- Best model checkpointing
- Per-class IoU reporting

The best model is saved as:

`best_deeplab.pth`

---

## 🖥️ Running the Analytics Dashboard

After training, launch the GUI:

```bash
python app.py
```

**Dashboard Features**

- Load and analyze images
- Overlay segmentation results
- Terrain composition donut chart
- Per-class confidence bar chart
- Inference latency tracking
- Performance history visualization

The dashboard automatically loads `best_deeplab.pth` if available.

---

## 📊 Evaluation Metrics

The system evaluates performance using:

- Mean Intersection over Union (mIoU)
- Pixel Accuracy
- Per-class IoU
- Average confidence per class

These metrics provide insight into segmentation quality and model reliability.

---

## ⚠️ Known Limitations

- DeepLabV3 is computationally intensive
- High GPU utilization is expected
- Real-time inference may be slow on low-end hardware
- Laptop GPUs may experience thermal throttling

These are expected behaviors for dense segmentation models.

---

## 🔮 Future Improvements

- Lightweight backbones (MobileNetV4, EfficientNet)
- Model export to ONNX / TensorRT
- Video stream segmentation
- ROS integration
- Real-time edge deployment
- Dataset augmentation pipeline

---

## 🤝 Contributing

Contributions are welcome.

You may contribute by:

- Improving model performance
- Adding deployment tools
- Extending the dashboard
- Optimizing memory usage
- Improving documentation

Fork the repository and submit a pull request.

---

## 📜 License

This project is released under the MIT License.

---

## 👨‍💻 Authors

Developed for academic and hackathon purposes.

Contributors:

- Manya Chawla
- Lakshya Jindal
- Kartikay Tayal
- Hriday Shah
