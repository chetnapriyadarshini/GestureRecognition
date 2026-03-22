# Gesture Recognition

A Jupyter Notebook implementing video-based hand gesture recognition for a smart TV remote control application, comparing two deep learning architectures — **CNN+RNN** and **3D Convolutional Network (Conv3D)** — to classify five distinct gesture commands from short video sequences.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Dataset](#dataset)
- [Architectures](#architectures)
- [Notebook Contents](#notebook-contents)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

---

## Overview

Smart TVs require hands-free control interfaces as an alternative to physical remotes. This project builds a gesture recognition model that continuously monitors a webcam stream and classifies five distinct hand gestures — each mapped to a TV command (volume up/down, channel change, pause, mute). The core challenge is modelling the **temporal dimension** of video: a gesture is not a single frame but a sequence of frames unfolding over time.

Two architectures are implemented and compared, each offering a different approach to spatiotemporal feature learning.

---

## Background

Video classification requires capturing both spatial features (what a frame looks like) and temporal features (how the video changes across frames). Two principal approaches exist:

**CNN + RNN:** A 2D CNN processes each frame independently, extracting a spatial feature vector. The sequence of feature vectors is then fed through a recurrent layer (GRU/LSTM) that models the temporal evolution. Transfer learning is applied to the CNN backbone to leverage pretrained ImageNet representations.

**Conv3D:** A 3D convolutional filter slides across both the spatial dimensions (height, width) and the temporal dimension (frames), capturing spatiotemporal patterns in a single unified operation. This approach treats a video clip as a 4D tensor and convolves over all three dimensions simultaneously.

---

## Dataset

The training dataset consists of several hundred short video clips (2–3 seconds each), each divided into a fixed sequence of **30 frames**. Videos are categorised into five gesture classes:

| Gesture | TV Command |
|---|---|
| Thumbs Up | Increase Volume |
| Thumbs Down | Decrease Volume |
| Left Swipe | Rewind / Previous Channel |
| Right Swipe | Fast Forward / Next Channel |
| Stop (palm) | Pause / Mute |

A **custom Python generator** handles data ingestion, frame-level preprocessing, and batch construction — enabling memory-efficient training on the full video dataset without loading all clips into RAM simultaneously.

---

## Architectures

### Architecture 1 — CNN + RNN (with Transfer Learning)

```
Video Clip (30 frames)
       │
  ┌────▼────────────────────────┐
  │  2D CNN (pretrained)        │  Per-frame spatial features
  │  (e.g. MobileNet / VGG)    │  Shape: (batch, frames, features)
  └────────────┬────────────────┘
               │
  ┌────────────▼────────────────┐
  │  GRU / LSTM Layer           │  Temporal modelling across frames
  └────────────┬────────────────┘
               │
  ┌────────────▼────────────────┐
  │  Dense + Softmax (5 classes)│
  └─────────────────────────────┘
```

### Architecture 2 — Conv3D

```
Video Clip: 4D tensor (100 × 100 × 30) × 3 channels
       │
  ┌────▼────────────────────────────┐
  │  3D Convolutional Layers        │  Spatiotemporal feature extraction
  │  Kernel slides over H × W × T  │  across height, width, and time
  └────────────┬────────────────────┘
               │
  ┌────────────▼────────────────────┐
  │  3D MaxPooling + BatchNorm      │
  └────────────┬────────────────────┘
               │
  ┌────────────▼────────────────────┐
  │  Flatten → Dense → Softmax      │  5-class gesture output
  └─────────────────────────────────┘
```

---

## Notebook Contents

| Section | Description |
|---|---|
| Data Loading & Generator | Custom generator for memory-efficient batch loading of video frames |
| Frame Preprocessing | Resizing, normalisation, and augmentation of individual frames |
| CNN+RNN Architecture | Transfer learning CNN backbone + GRU temporal layer |
| Conv3D Architecture | 3D convolutional network for direct spatiotemporal learning |
| Training & Callbacks | ModelCheckpoint, ReduceLROnPlateau, EarlyStopping |
| Architecture Comparison | Accuracy, loss, training time, and parameter count comparison |
| Write-Up | `Gesture_Recognition_WriteUp.docx` — detailed analysis of design choices and results |

---

## Saved Model

The repository includes a pre-trained model checkpoint:

```
model-00014-4.91413-0.99095-5.54131-0.67000.h5
         │       │              │
       Epoch  Train Acc     Val Acc (67%)
               (99%)
```

The filename encodes epoch number, training loss, training accuracy, validation loss, and validation accuracy — indicating **99% training accuracy** achieved by epoch 14, with validation accuracy at 67%, reflecting expected generalisation gap on the small dataset.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | CNN, GRU, Conv3D model definition and training |
| `numpy` | Frame array manipulation and batch construction |
| `opencv` / `PIL` | Frame-level image loading and preprocessing |
| `matplotlib` | Training curve visualisation |

---

## Setup and Installation

```bash
git clone https://github.com/chetnapriyadarshini/GestureRecognition.git
cd GestureRecognition
pip install tensorflow numpy opencv-python matplotlib pillow
jupyter notebook Gesture_recognition_Final.ipynb
```

> **Note:** A GPU runtime is strongly recommended. The pretrained model checkpoint (`.h5` file) can be loaded directly for inference without retraining.

---

## Results

The CNN+RNN architecture with transfer learning converges faster and achieves higher validation accuracy compared to the Conv3D model on this dataset size, owing to the benefit of pretrained ImageNet features. The Conv3D model captures richer spatiotemporal patterns but requires more data to generalise effectively. Detailed comparative analysis is available in `Gesture_Recognition_WriteUp.docx`.

---

## References

- Donahue, J. et al. (2015). *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*. CVPR 2015.
- Tran, D. et al. (2015). *Learning Spatiotemporal Features with 3D Convolutional Networks*. ICCV 2015.

---

## Contact

Created by [@chetnapriyadarshini](https://github.com/chetnapriyadarshini) — feel free to reach out with questions or suggestions.
