<p align="center">
  <h1 align="center">🚗 Child Presence Detection (CPD) — YOLOv8n Classifier</h1>
  <p align="center">
    <em>Preventing hot-car tragedies with edge-deployable deep learning</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/YOLOv8n--cls-Ultralytics-purple?logo=pytorch" alt="YOLOv8n-cls">
  <img src="https://img.shields.io/badge/FPGA-PYNQ%20Z2-green?logo=xilinx" alt="PYNQ Z2">
  <img src="https://img.shields.io/badge/GPU-RTX%204060-76B900?logo=nvidia" alt="RTX 4060">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="MIT License">
</p>

---

## 📋 Overview

This project trains a **YOLOv8n classification** model to detect the presence (or absence) of children inside a vehicle cabin. The goal is to power an **in-cabin child presence detection (CPD)** system that can trigger alerts when a child is left unattended — helping to prevent heat-stroke fatalities.

The system classifies each camera frame into **3 classes** and is designed to be small enough for deployment on an **FPGA (PYNQ Z2)** via the Xilinx FINN framework.

### 🎯 Why Classification, Not Detection?

For this safety-critical use case we only need to know **whether** a child is present, not **where** exactly. A lightweight classification head is:
- Smaller & faster than a detection head
- Easier to quantise for FPGA deployment
- Sufficient for the binary alert decision

---

## 🏷️ Class Definitions

| Class | Label | Description | Action |
|:---:|:---|:---|:---|
| 0 | `empty` | No child present — empty seats, objects, or adults only | ✅ Safe |
| 1 | `adult_child` | Adult(s) **and** child(ren) present together (supervised) | ✅ Safe |
| 2 | `child_only` | Child(ren) present **WITHOUT** adult supervision | 🚨 **DANGER — Trigger Alert!** |

---

## 📊 Dataset

### SVIRO (Synthetic Vehicle Interior Rear-seat Occupancy)

We use the [SVIRO dataset](https://sviro.kIT.edu) — a large-scale synthetic dataset of vehicle interior images across 10 car models with 7 occupancy classes.

#### 7-Class → 3-Class Mapping

| Original SVIRO Classes | → | CPD Class |
|:---|:---:|:---|
| `empty_seat` | → | `empty` |
| `everyday_object` | → | `empty` |
| `adult_only` | → | `empty` |
| `adult_and_child` | → | `adult_child` |
| `child_in_seat_with_adult` | → | `adult_child` |
| `child_in_seat` | → | `child_only` |
| `child_not_in_seat` | → | `child_only` |

#### Dataset Layout (after preparation)

```
data/sviro_yolo/
├── train/
│   ├── empty/          # ~X,XXX images
│   ├── adult_child/    # ~X,XXX images
│   └── child_only/     # ~X,XXX images
├── val/
│   ├── empty/
│   ├── adult_child/
│   └── child_only/
└── test/
    ├── empty/
    ├── adult_child/
    └── child_only/
```

---

## ⚙️ Setup

### Prerequisites

- **Python** 3.8+
- **NVIDIA GPU** with CUDA support (tested on RTX 4060, 8 GB VRAM)
- **CUDA Toolkit** 11.8+ and **cuDNN**

### Installation

```bash
# Clone the repository
git clone <your-repo-url> cpd_yolo
cd cpd_yolo

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision
onnx>=1.14.0
onnxruntime>=1.15.0
matplotlib
numpy
Pillow
```

---

## 🚀 Usage

### 1️⃣ Prepare Dataset

Download SVIRO and run the preparation script to remap 7 classes → 3 classes and organise into train/val/test splits:

```bash
python scripts/prepare_dataset.py --source data/sviro_raw --output data/sviro_yolo
```

### 2️⃣ Train Model

**Two-stage training** (recommended):

```bash
python scripts/train.py --data data/sviro_yolo --batch 16 --device 0
```

This runs:
- **Stage 1**: Freeze backbone (first 10 layers), 50 epochs, `lr=0.01`
- **Stage 2**: Unfreeze all, 30 epochs, `lr=0.001`

**Single-stage training** (simpler):

```bash
python scripts/train.py --single-stage --epochs-stage1 80 --epochs-stage2 0
```

**Resume from checkpoint**:

```bash
python scripts/train.py --resume runs/classify/stage1/weights/last.pt
```

### 3️⃣ Evaluate Model

```bash
python scripts/evaluate.py \
    --model runs/classify/train/weights/best.pt \
    --data data/sviro_yolo/test \
    --output results/
```

Outputs:
- `results/confusion_matrix.png`
- `results/sample_predictions.png`
- `results/evaluation_report.txt`

### 4️⃣ Export Model

```bash
python scripts/export_model.py \
    --model runs/classify/train/weights/best.pt \
    --imgsz 640 \
    --output-dir exports/
```

Exports:
- `exports/best.onnx` — for FPGA deployment
- `exports/best.torchscript` — backup

---

## 📁 Project Structure

```
cpd_yolo/
├── 📄 README.md                  # This file
├── 📄 requirements.txt           # Python dependencies
├── 📄 best_model_path.txt        # Auto-generated: path to best weights
│
├── 📂 scripts/
│   ├── train.py                  # Two-stage training script
│   ├── evaluate.py               # Evaluation & metrics
│   ├── export_model.py           # ONNX / TorchScript export
│   └── prepare_dataset.py        # SVIRO → 3-class dataset prep
│
├── 📂 data/
│   └── sviro_yolo/               # Prepared dataset
│       ├── train/
│       ├── val/
│       └── test/
│
├── 📂 runs/
│   └── classify/                 # Training outputs
│       ├── stage1/               # Stage 1 checkpoints
│       ├── stage2/               # Stage 2 checkpoints
│       └── train/                # Final best model
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── 📂 results/                   # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── evaluation_report.txt
│
└── 📂 exports/                   # Exported models
    ├── best.onnx
    └── best.torchscript
```

---

## 🔧 Training Details

### Two-Stage Transfer Learning

| | Stage 1 | Stage 2 |
|:---|:---|:---|
| **Base model** | `yolov8n-cls.pt` (pretrained) | Stage 1 best weights |
| **Frozen layers** | First 10 (backbone) | None (all trainable) |
| **Learning rate** | `0.01` | `0.001` |
| **Epochs** | 50 | 30 |
| **Early stopping** | Patience = 30 | Patience = 30 |

### Hyperparameters

| Parameter | Value | Rationale |
|:---|:---|:---|
| Image size | 640 × 640 | Balances resolution vs. speed |
| Batch size | 16 | Conservative for 8 GB VRAM |
| Optimizer | SGD (YOLO default) | Stable convergence |
| Workers | 8 | Parallel data loading |

### ⏱️ Estimated Training Time (RTX 4060)

| Stage | Epochs | Est. Time |
|:---|:---|:---|
| Stage 1 | 50 | ~15–25 min |
| Stage 2 | 30 | ~10–15 min |
| **Total** | **80** | **~25–40 min** |

*Times vary with dataset size and augmentation settings.*

---

## 🔌 PYNQ Z2 / FPGA Deployment

### Next Steps for Edge Deployment

1. **Quantisation-Aware Training (QAT)**
   - Re-train with INT8/INT4 quantisation using Brevitas
   - Target model size: < 1 MB

2. **FINN Compiler Pipeline**
   - Convert quantised ONNX → FINN-ONNX → HLS → Bitstream
   - Estimate resource utilisation on Zynq-7020

3. **PYNQ Overlay**
   - Deploy as a PYNQ overlay with DMA-based frame input
   - Target latency: < 50 ms per frame at 640×640

4. **System Integration**
   - USB camera → FPGA inference → Alert GPIO/buzzer
   - Continuous monitoring with watchdog timer

### Deployment Architecture

```
Camera (USB) → Frame Buffer → FPGA Accelerator → Classification
                                    │
                          ┌─────────┴─────────┐
                          │                    │
                     child_only?          safe (empty
                          │              / adult_child)
                          ▼
                    🚨 ALERT SYSTEM
                   (Buzzer / GSM / App)
```

---

## 🖥️ Hardware

| Component | Specification |
|:---|:---|
| **Training GPU** | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| **Training OS** | Windows 10/11 |
| **Target FPGA** | Digilent PYNQ-Z2 (Xilinx Zynq-7020) |
| **Target Camera** | USB webcam or MIPI CSI |

---

## 📚 References

1. **YOLOv8** — Ultralytics. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. **SVIRO Dataset** — Dias Da Cruz, S., Wasenmuller, O., Beez, H.-P., & Stricker, D. (2020). *SVIRO: Synthetic Vehicle Interior Rear Seat Occupancy Dataset and Benchmark.* [https://sviro.kIT.edu](https://sviro.kIT.edu)
3. **FINN Framework** — Xilinx Research Labs. [https://github.com/Xilinx/finn](https://github.com/Xilinx/finn)
4. **PYNQ** — Xilinx. [https://pynq.io](https://pynq.io)
5. **Brevitas** — Xilinx. [https://github.com/Xilinx/brevitas](https://github.com/Xilinx/brevitas)

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>⚠️ This system is designed as a supplementary safety measure and should NOT be the sole safeguard for child safety in vehicles.</strong>
</p>

<p align="center">
  Made with ❤️ for child safety
</p>
