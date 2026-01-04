# AI-Powered Satellite Vision System

**100% Pure Deep Learning - Zero Classical Methods**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app_advanced.py
```

Open browser at: `http://localhost:8501`

---

## 📋 Features

### ✨ Pure Deep Learning Enhancement
- **Shadow Removal**: Swin2SR Transformer (Hugging Face)
- **Haze Removal**: Swin2SR Transformer (Hugging Face)
- **Object Detection**: Faster R-CNN (PyTorch)
- **AI Captions**: Natural language scene descriptions

### 🔍 AI Scene Understanding
- Water tanks and storage facilities
- Towers and vertical structures
- Buildings and infrastructure
- Roads and pathways
- Vegetation coverage analysis

### 🔄 Dual-Image Mode
- Change detection between time periods
- Structural changes quantified
- Vegetation growth/clearance tracking
- Construction/demolition detection

---

## 🛠️ System Requirements

### Minimum
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB minimum, 16GB recommended
- **Internet**: Required for first-time model download
- **Python**: 3.8 or higher

### Dependencies
- PyTorch (Deep Learning framework)
- Transformers (Hugging Face models)
- OpenCV (Image I/O only)
- Streamlit (Web UI)
- NumPy, Pillow (Scientific computing)

---

## 📖 How to Use

### Single Image Mode
1. Upload satellite image (JPG, PNG)
2. Click "🚀 Enhance Image"
3. Wait 5-15 seconds (deep learning processing)
4. View enhanced image and AI analysis

### Dual Image Mode
1. Upload two images (before/after)
2. Click "🔄 Compare & Detect Changes"
3. Wait 10-25 seconds
4. View comparison and change description

---

## 🌐 Deep Learning Models

### Swin2SR (Image Enhancement)
- **Source**: Hugging Face (`caidas/swin2SR-classical-sr-x2-64`)
- **Type**: Transformer-based super-resolution
- **Purpose**: Shadow/haze removal, image enhancement
- **Speed**: 5-15 seconds per image

### Faster R-CNN (Object Detection)
- **Source**: PyTorch/TorchVision
- **Type**: Region-based CNN
- **Purpose**: Detect objects in satellite imagery
- **Speed**: 1-2 seconds per image

---

## 📁 Project Structure

```
d:\Clg_project\
├── app_advanced.py              # Streamlit UI
├── inference_advanced.py        # Processing engine
├── dl_models.py                 # DL orchestrator
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── core_modules/
│   ├── shadow_remover.py       # Swin2SR shadow removal
│   ├── haze_remover.py         # Swin2SR haze removal
│   ├── scene_interpreter.py    # Faster R-CNN detection
│   ├── ai_captioner.py         # AI captions
│   ├── scene_analyzer.py       # (Disabled - pure DL mode)
│   └── cloud_remover.py        # (Disabled - pure DL mode)
└── models/                      # (Auto-downloaded from Hugging Face)
```

---

## 🔧 Troubleshooting

### Slow Processing
- **Issue**: Takes 5-15 seconds per image
- **Reason**: Swin2SR is a heavy transformer model
- **Solution**: This is normal for pure deep learning

### Model Download
- **Issue**: First run takes time
- **Reason**: Downloading Swin2SR weights (~200MB)
- **Solution**: Wait for download to complete

### Import Errors
- **Issue**: `ModuleNotFoundError`
- **Solution**: Run `pip install -r requirements.txt`

---

## 📊 Performance

- **Single Image**: 5-15 seconds (with Swin2SR)
- **Dual Image**: 10-25 seconds
- **Model Loading**: 3-5 seconds (first time only)

---

## 🎯 Technical Details

### Pure Deep Learning Stack
- **No classical CV methods** (no CLAHE, no bilateral filter, no Sobel)
- **100% PyTorch/Transformers**
- **Pre-trained models only**

### Processing Pipeline
```
Input Image
    ↓
Swin2SR Transformer (Enhancement)
    ↓
Faster R-CNN (Object Detection)
    ↓
AI Caption Generation
    ↓
Enhanced Output + Analysis
```

---

## 📝 License

This project uses open-source models:
- PyTorch (BSD License)
- Transformers (Apache 2.0)
- Hugging Face Models (Various open licenses)

---

**Built with 100% pure deep learning models** 🤖
