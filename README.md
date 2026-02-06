[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) [![TorchVision](https://img.shields.io/badge/TorchVision-0.15+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/vision/) [![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/) [![Pillow](https://img.shields.io/badge/Pillow-9.5+-C5C5C5?style=flat-square&logo=python&logoColor=black)](https://python-pillow.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat-square&logo=plotly&logoColor=white)](https://matplotlib.org/) [![PyQt6](https://img.shields.io/badge/PyQt6-6.4+-41CD52?style=flat-square&logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/) [![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/) [![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit) [![MPS](https://img.shields.io/badge/MPS-Supported-000000?style=flat-square&logo=apple&logoColor=white)](https://developer.apple.com/metal/) [![DeepLabV3](https://img.shields.io/badge/Model-DeepLabV3-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://arxiv.org/abs/1706.05587) [![MobileNetV3](https://img.shields.io/badge/Backbone-MobileNetV3-34A853?style=flat-square&logo=google&logoColor=white)](https://arxiv.org/abs/1905.02244) [![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
# 🌵 DesertMind

**Off-Road Semantic Segmentation & Analytics System**

> *Empowering autonomous navigation in unstructured desert environments through advanced semantic segmentation.*

DesertMind is a production-grade deep learning system for pixel-level terrain analysis in off-road environments. Built with DeepLabV3 and MobileNetV3, it combines GPU-optimized training pipelines with an interactive analytics dashboard for real-time terrain understanding.

📌 Production Model: This project uses a three-phase experimental approach. The final production model is from Phase 2 (fine-tuned with augmentation, achieving 0.507 mIoU). Phase 1 establishes the baseline, while Phase 3 demonstrates ablation studies.

---
## 📋 Table of Contents

- [Key Features](#-key-features)
- [Motivation](#-motivation)
- [System Architecture](#️-system-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset Format](#-dataset-format)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training Pipeline](#training-pipeline)
  - [Analytics Dashboard](#analytics-dashboard)
- [Experimental Results](#-experimental-results)
- [Model Architecture](#-model-architecture)
- [Performance Metrics](#-performance-metrics)
- [Limitations & Future Work](#️-limitations--future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
---

## ✨ Key Features

### Core Capabilities
- **Advanced Segmentation Model**: DeepLabV3 with MobileNetV3 Large backbone for lightweight, high-accuracy segmentation
- **Multi-Phase Training Pipeline**: Systematic experimentation with baseline training, fine-tuning, and ablation studies
- **Interactive Analytics Dashboard**: PyQt6-based GUI with real-time visualization and performance monitoring
- **Production-Ready Inference**: Optimized for low-latency predictions with device-agnostic support (CUDA/MPS/CPU)

### Engineering Features
- **Mixed Precision Training**: AMP-enabled for faster training and reduced memory usage
- **Data Augmentation**: Random flips, brightness/contrast adjustments for robust generalization
- **Early Stopping & Checkpointing**: Automatic best-model saving based on validation mIoU
- **Comprehensive Metrics**: Mean IoU (mIoU), Pixel Accuracy, Per-Class IoU, and confidence analysis
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Terrain Classes (10 Categories)
```
Sky | Sand | Dry Grass | Trees | Bushes | Rocks | Logs | Ground Clutter | Lush Bushes | Background
```

---

## 🎯 Motivation

Off-road environments present unique challenges for computer vision systems:

**Challenges:**
- ❌ Lack of structured cues (no lane markings or clear boundaries)
- ❌ Variable lighting and unpredictable textures
- ❌ Traditional models struggle with unstructured terrain
- ❌ High variability in sand, rocks, vegetation composition

**DesertMind Solution:**
- ✅ Pixel-level semantic understanding of terrain
- ✅ Robust to lighting and texture variations
- ✅ Real-time inference for autonomous decision-making
- ✅ Comprehensive analytics for mission planning

**Applications:** Autonomous vehicles, drones, robotics, exploration systems, and terrain analysis.

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Input Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ← Resize, Normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MobileNetV3     │ ← Feature Extraction
│ Backbone        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DeepLabV3       │ ← Atrous Spatial Pyramid Pooling
│ ASPP Module     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Segmentation    │ ← 10-Class Classifier
│ Head            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prediction Map  │ ← Pixel-wise Classes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization & │
│   Analytics     │
└─────────────────┘
```

---

## 📂 Repository Structure

```
desertmind/
│
├── 📄 README.md                                    # Project documentation
├── 📄 OUTPUTS.txt                                  # Detailed training logs & analysis
├── 📄 requirements.txt                             # Python dependencies
├── 📄 LICENSE                                      # MIT License
│
├── 🖥️  app.py                                      # PyQt6 analytics dashboard
│
├── web_app.py                                      # website for any user to test
|
|
├── 📊 Dataset Loaders
│   ├── dataset_phase1.py                          # Baseline dataset (no augmentation)
│   ├── dataset_finetune_augmentation_phase2.py    # Augmented dataset (Phase 2)
│   └── dataset_weighted_classes_phase3.py         # Cropped + weighted dataset (Phase 3)
│
├── 🏋️ Training Scripts
│   ├── train_deeplab_phase1.py                    # Phase 1: Baseline training
│   ├── train_deeplab_finetune_phase2.py           # Phase 2: Fine-tuning (PRODUCTION)
│   └── train_deeplab_finetune_weightedClass_phase3 # Phase 3: Experimental optimization
│
└── 💾 Model Checkpoints
    └── best_deeplab.pth                            # Pre-trained model weights
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main GUI application for inference and real-time analytics |
| `dataset_*.py` | Custom PyTorch datasets with varying augmentation strategies |
| `train_*.py` | Training scripts implementing different experimental phases |
| `OUTPUTS.txt` | Comprehensive logs, results, and engineering insights |
| `requirements.txt` | All Python package dependencies |
| `web_app.py` | open website for inference and real-time analytics |

---

## 📁 Dataset Format

The project expects the **Offroad Segmentation Training Dataset** in the following structure:

```
Offroad_Segmentation_Training_Dataset/
│
├── train/
│   ├── Color_Images/           # RGB images (PNG format)
│   └── Segmentation/           # Ground-truth masks (PNG, raw labels)
│
├── val/
│   ├── Color_Images/
│   └── Segmentation/
│
└── test/                       # Optional test set
    ├── Color_Images/
    └── Segmentation/
```

### Dataset Specifications

- **Format**: PNG images and masks
- **Naming**: Masks must match corresponding image filenames
- **Label Encoding**: Raw mask values are remapped to training classes (0-9)
- **Resolution**: Variable (resized to 520×520 during training)

### Label Mapping

| Raw Value | Class ID | Terrain Type |
|-----------|----------|--------------|
| 100 | 0 | Sky |
| 200 | 1 | Sand/Landscape |
| 300 | 2 | Dry Grass |
| 500 | 3 | Trees |
| 550 | 4 | Bushes |
| 600 | 5 | Rocks |
| 700 | 6 | Logs |
| 800 | 7 | Ground Clutter |
| 7100 | 8 | Lush Bushes |
| 10000 | 9 | Background |

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- NVIDIA GPU with CUDA support (recommended for training)
- 8GB+ RAM (16GB recommended for training)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ktayal2006/desertmind.git
   cd desertmind
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
   ```

### Dependencies

```
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Vision models and transforms
numpy>=1.24.0         # Numerical operations
PyQt6>=6.4.0          # GUI framework
Pillow>=9.5.0         # Image processing
matplotlib>=3.7.0     # Visualization
```

---

## 💻 Usage

### Training Pipeline

DesertMind implements a **three-phase experimental approach** for systematic model development:

#### **Phase 1: Baseline Training** 🏁

Establish a strong baseline without fine-tuning or augmentations.

```bash
python train_deeplab_phase1.py
```

**Configuration:**
- Loss Function: Cross-Entropy
- Learning Rate: 1e-4
- Epochs: 10
- Batch Size: 2
- Early Stopping Patience: 2

**Before Running:**
Update the dataset path in `train_deeplab_phase1.py`:
```python
ROOT = r"path/to/Offroad_Segmentation_Training_Dataset"
```

**Expected Output:**
```
Using device: cuda
Train size: 2857 | Val size: 317

Epoch 1/10 | train loss 0.8234 | val loss 0.7456 | val mIoU 0.3912
Epoch 5/10 | train loss 0.4892 | val loss 0.4567 | val mIoU 0.4669
Epoch 9/10 | train loss 0.3421 | val loss 0.3892 | val mIoU 0.4962 ⭐
Saved best model: 0.4962
```

**Key Results:**
- ✅ Best Validation mIoU: **0.496**
- ✅ No overfitting observed
- ✅ Stable convergence

---

#### **Phase 2: Fine-Tuning** 🎯 **(PRODUCTION MODEL)**

Refine the model with layer-wise learning rates and augmentation.

```bash
python train_deeplab_finetune_phase2.py
```

**Configuration:**
- Loads Phase 1 checkpoint
- Learning Rate: 3e-5
- Epochs: 6
- Augmentation: Horizontal flip, brightness/contrast adjustment
- Per-Class IoU tracking enabled

**Two-Stage Approach:**
1. **Stage 1**: Freeze backbone, train head only
2. **Stage 2**: Unfreeze backbone with differential learning rates

**Expected Output:**
```
Loaded best checkpoint for fine-tuning
Backbone frozen

Epoch 1/6 | val mIoU 0.5056 | pixel acc 0.8523
Epoch 2/6 | val mIoU 0.5070 ⭐ | pixel acc 0.8547
Early stopping triggered.

Per-class IoU:
  Class 0 (Sky): 0.9712
  Class 1 (Sand): 0.6234
  Class 5 (Rocks): 0.4156
  ...
```

**Key Results:**
- ✅ Best Validation mIoU: **0.507**
- ✅ Pixel Accuracy: **~85.5%**
- ✅ Production-ready checkpoint

---

#### **Phase 3: Experimental Optimization** 🧪

Test aggressive techniques: class weighting, Dice loss, random cropping.

```bash
python train_deeplab_finetune_weightedClass_phase3
```

**Configuration:**
- Weighted Cross-Entropy + Dice Loss
- Random 512×512 crops
- Differential learning rates (backbone vs. head)
- Class weighting based on inverse frequency

**Expected Output:**
```
Class weights: [1.02, 0.87, 1.34, 2.41, 1.89, 1.56, 2.98, 1.67, 2.23, 0.95]

Epoch 1/12 | val mIoU 0.4823 | pixel acc 0.8134
Epoch 6/12 | val mIoU 0.4867 ⭐ | pixel acc 0.8156
```

**Key Results:**
- ⚠️ Validation mIoU: **0.487** (regression)
- ⚠️ Demonstrates limitations of over-optimization
- ✅ Valuable ablation study for understanding model ceiling

---

### Analytics Dashboard

Launch the interactive GUI for real-time inference and visualization:

```bash
python app.py
```

#### Dashboard Features

| Component | Description |
|-----------|-------------|
| **Terrain Segmentation View** | Side-by-side display of original image and segmentation overlay |
| **Terrain Composition** | Donut chart showing pixel distribution across 10 classes |
| **Model Confidence** | Bar chart of average confidence per terrain class |
| **Performance Trend** | Line chart tracking inference latency over time |
| **Terrain Color Key** | Visual legend mapping colors to terrain classes |

#### Usage Workflow

1. Click **"IMPORT & ANALYZE OFF-ROAD SCENE"**
2. Select an off-road image (PNG/JPG format)
3. Wait for processing (typically <500ms on GPU)
4. Explore visualizations:
   - Segmentation overlay with 55% alpha blending
   - Class distribution statistics
   - Per-class confidence scores
   - Inference latency monitoring

#### Example Metrics

```
Inference Time: 342ms (CUDA)
Dominant Terrain: Sand (42.3%)
Average Confidence: 0.87
Top Classes:
  • Sand: 42.3% (conf: 0.91)
  • Sky: 28.7% (conf: 0.97)
  • Rocks: 15.2% (conf: 0.78)
```

---

## 📊 Experimental Results

### Performance Summary

| Phase | Configuration | Val mIoU | Pixel Acc | Status |
|-------|--------------|----------|-----------|--------|
| **Phase 1** | Baseline (no fine-tuning) | 0.496 | - | ✅ Baseline |
| **Phase 2.1** | Head-only fine-tuning | 0.502 | 0.855 | ✅ Incremental |
| **Phase 2.2** | Backbone fine-tuning | **0.507** | **0.855** | ⭐ **PRODUCTION** |
| **Phase 3** | Weighted + Dice + Cropping | 0.487 | 0.815 | ⚠️ Regression |

### Per-Class Performance (Phase 2 - Best Model)

| Class | IoU | Notes |
|-------|-----|-------|
| Sky (0) | 0.971 | Saturated performance |
| Sand (1) | 0.623 | Strong representation |
| Dry Grass (2) | 0.512 | Moderate |
| Trees (3) | 0.456 | Challenging due to occlusion |
| Bushes (4) | 0.389 | Texture variation |
| Rocks (5) | 0.416 | Size variability |
| Logs (6) | 0.243 | Rare class |
| Ground Clutter (7) | 0.278 | High intra-class variation |
| Lush Bushes (8) | 0.167 | Underrepresented |
| Background (9) | 0.982 | Well-defined |

### Key Insights

**What Worked:**
- ✅ Backbone fine-tuning with differential learning rates
- ✅ Data augmentation (flips, brightness/contrast)
- ✅ Mixed precision training for efficiency
- ✅ Early stopping prevents overfitting

**What Didn't Work:**
- ❌ Aggressive class weighting → shifted objective, harmed dominant classes
- ❌ Dice loss in multi-class setting → gradient instability
- ❌ Random cropping → removed global context (sky/horizon confusion)

**Engineering Takeaway:**
> The model has reached its **data ceiling** (~0.51 mIoU). Further gains require higher-quality annotations, more diverse data, or architectural changes.

For detailed analysis, see [`OUTPUTS.txt`](OUTPUTS.txt).

---

## 🧠 Model Architecture

### DeepLabV3 with MobileNetV3 Large

```python
Model(
  backbone: MobileNetV3Large(
    # Inverted Residual Blocks
    # Efficient feature extraction
    # Pre-trained on ImageNet
  ),
  aspp: ASPP(
    # Atrous Spatial Pyramid Pooling
    # Multi-scale context aggregation
    # Dilation rates: [6, 12, 18]
  ),
  classifier: Conv2d(256 → 10 classes)
)
```

### Architecture Highlights

| Component | Configuration |
|-----------|--------------|
| **Backbone** | MobileNetV3 Large (pre-trained) |
| **ASPP Module** | Atrous rates: 6, 12, 18 |
| **Output Stride** | 8 (high resolution) |
| **Classifier** | 256 → 10 classes |
| **Auxiliary Loss** | Enabled during training |
| **Parameters** | ~8.5M (lightweight) |

### Training Configuration

```python
Loss: CrossEntropyLoss (with optional class weights)
Optimizer: AdamW
Learning Rate:
  - Backbone: 1e-5 (fine-tuning)
  - Head: 3e-5 to 5e-5
Scheduler: ReduceLROnPlateau (mode='max')
Mixed Precision: AMP (CUDA)
Batch Size: 2 (memory-constrained)
Image Size: 520×520 (training), variable (inference)
```

---

## 📈 Performance Metrics

### Evaluation Metrics

1. **Mean Intersection over Union (mIoU)**
   ```
   mIoU = (1/N) Σ (TP / (TP + FP + FN))
   ```
   Primary metric for segmentation quality.

2. **Pixel Accuracy**
   ```
   Accuracy = (Correct Pixels) / (Total Pixels)
   ```
   Overall correctness across all pixels.

3. **Per-Class IoU**
   Individual class performance for debugging.

4. **Model Confidence**
   Average softmax probability per class.

### Inference Performance

| Hardware | Device | Avg. Latency | Throughput |
|----------|--------|--------------|------------|
| RTX 3050 | CUDA | 342ms | ~3 FPS |
| M1 Pro | MPS | 580ms | ~1.7 FPS |
| CPU (Intel i7) | CPU | 1850ms | ~0.5 FPS |

*Tested on 520×520 images*

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Compute Requirements**: DeepLabV3 is resource-intensive (RTX 3050 minimum recommended)
- **Real-Time Constraints**: Not suitable for real-time edge deployment without optimization
- **Class Imbalance**: Rare classes (logs, lush bushes) underperform
- **Data Ceiling**: Model performance saturates at ~0.51 mIoU with current dataset

### Future Improvements

**Model Optimization:**
- [ ] Export to ONNX/TensorRT for faster inference
- [ ] Experiment with lighter backbones (MobileNetV4, EfficientNet-Lite)
- [ ] Implement knowledge distillation for edge deployment

**Data & Training:**
- [ ] Expand dataset with diverse lighting/weather conditions
- [ ] Implement focal loss for hard examples
- [ ] Multi-scale training and inference

**Deployment:**
- [ ] ROS (Robot Operating System) integration
- [ ] Video stream processing pipeline
- [ ] Real-time visualization on embedded devices

**Dashboard:**
- [ ] Add batch processing mode
- [ ] Export segmentation results (masks, statistics)
- [ ] Compare multiple models side-by-side

---

## 🤝 Contributing

Contributions are welcome! This project is designed for collaborative improvement.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** (code, documentation, experiments)
4. **Test thoroughly**
   ```bash
   # Run training script on sample data
   python train_deeplab_phase1.py
   
   # Test dashboard
   python app.py
   ```
5. **Submit a pull request** with a clear description

### Contribution Ideas

- 🔧 Add new augmentation techniques
- 🚀 Implement model export (ONNX, CoreML)
- 📊 Enhance dashboard with new visualizations
- 📝 Improve documentation and tutorials
- 🧪 Conduct new ablation studies

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2024 DesertMind Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👨‍💻 Authors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ktayal2006">
        <img src="https://github.com/ktayal2006.png" width="100px;" alt="Kartikay Tayal"/>
        <br />
        <sub><b>Kartikay Tayal</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/manyachawla22">
        <img src="https://github.com/manyachawla22.png" width="100px;" alt="Manya Chawla"/>
        <br />
        <sub><b>Manya Chawla</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/lakshya1907">
        <img src="https://github.com/lakshya1907.png" width="100px;" alt="Lakshya Jindal"/>
        <br />
        <sub><b>Lakshya Jindal</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Hridayshah18">
        <img src="https://github.com/Hridayshah18.png" width="100px;" alt="Hriday Shah"/>
        <br />
        <sub><b>Hriday Shah</b></sub>
      </a>
    </td>
  </tr>
</table>

### Acknowledgments

- **PyTorch Team** for the deep learning framework
- **TorchVision Contributors** for pre-trained models and utilities
- **Open-Source Community** for inspiration and tools

---
</div>
