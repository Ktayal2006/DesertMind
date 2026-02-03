![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green?logo=nvidia)
![GPU](https://img.shields.io/badge/GPU-RTX%203050-76B900?logo=nvidia)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-purple)
![Backbone](https://img.shields.io/badge/Model-DeepLabV3-orange)

# 🌵 DesertMind  
### Off-Road Semantic Scene Segmentation with DeepLabV3 (GPU-Optimized)

DesertMind is a **GPU-accelerated semantic segmentation pipeline** for **off-road environments**, built using **DeepLabV3 + ResNet-50** and trained on a custom off-road dataset.

This project focuses on:
- pixel-accurate terrain understanding
- real-world GPU constraints (RTX 3050, 4GB VRAM)
- stable, production-style PyTorch training
- clean, reproducible experimentation

If you’re interested in **autonomous navigation**, **off-road robotics**, or **scene understanding under harsh terrain**, this project is for you.

---

## 🚀 Key Features

- ✅ **DeepLabV3 + ResNet-50** semantic segmentation
- ✅ **CUDA + AMP (mixed precision)** training
- ✅ **Early stopping** (no wasted epochs)
- ✅ **GPU-safe configuration** for low-VRAM laptops
- ✅ **Windows-compatible multiprocessing**
- ✅ **Clean project structure**
- ✅ **Ready for real datasets, not toy demos**

---

## 🧠 What This Project Solves

Off-road environments are chaotic:
- no lane markings
- unstructured terrain
- dirt, sand, rocks, vegetation
- lighting and texture variation

DesertMind learns **pixel-level terrain semantics** so higher-level systems (planners, controllers, robots) can make decisions based on **what the ground actually is**.

---

## 🗂 Project Structure
desertmind_project/
├── dataset.py # Custom Dataset + label remapping
├── train_deeplab.py # GPU-optimized DeepLabV3 training script
├── best_deeplab.pth # Best saved model (after training)
├── README.md # This file
└── venv/ # Python virtual environment


Dataset directory (can live anywhere):
Offroad_Segmentation_Training_Dataset/
├── train/
│ ├── Color_Images/
│ └── Segmentation/
├── val/
│ ├── Color_Images/
│ └── Segmentation/
└── test/ (optional)


---

## 🧪 Model Details

- **Architecture:** DeepLabV3
- **Backbone:** ResNet-50 (ImageNet pretrained)
- **Loss:** Cross-Entropy
- **Metric:** Mean Intersection-over-Union (mIoU)
- **Classes:** 10 (custom off-road label set)

---

## ⚙️ Training Setup (Real-World)

This project was trained on:

- **GPU:** NVIDIA RTX 3050 (Laptop, 4GB VRAM)
- **OS:** Windows
- **Python:** 3.11
- **Framework:** PyTorch + TorchVision
- **CUDA:** Enabled
- **Precision:** Mixed Precision (AMP)

⚠️ The code is explicitly optimized for **low-VRAM GPUs**.

---

## 🔥 Performance Optimizations

To make DeepLabV3 usable on limited hardware, DesertMind includes:

- Mixed precision (`torch.amp`)
- cuDNN autotuning
- Reduced batch size
- Persistent DataLoader workers
- Early stopping (patience-based)
- GPU-safe memory usage

This is **not a notebook toy** — it’s a real training pipeline.

---


💾 Output

best_deeplab.pth → best model checkpoint (by validation mIoU)
Console logs → training + validation metrics per epoch

🧠 Why Early Stopping?

Instead of guessing epoch counts, DesertMind uses patience-based early stopping:
Trains up to a max epoch count
Stops automatically when learning plateaus
Saves GPU time and heat
Prevents overfitting

⚠️ Known Constraints

DeepLabV3 is computationally heavy
Full-resolution segmentation is slow on laptop GPUs
High GPU utilization is expected
Sustained temps above ~85°C should be avoided
This is expected behavior, not a bug.

🚀 Future Improvements

Lighter backbone (MobileNet / HRNet)
Input resolution scaling
Inference & visualization scripts
Export to ONNX / TensorRT
Integration with robotics stacks

🤝 Contributing
Pull requests, experiments, and improvements are welcome.
If you’re experimenting with:
off-road robotics
terrain classification
GPU-efficient segmentation
feel free to fork and build on it.

---

## 🏁 Installation

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
pip install torch torchvision numpy pillow matplotlib
ROOT = r"/path/to/Offroad_Segmentation_Training_Dataset"

Run training:
python train_deeplab.py

Expected output:

Using device: cuda
Train size: 2857 Val size: 317
Epoch 1/20 | train loss ... | val loss ... | val mIoU ...


---
