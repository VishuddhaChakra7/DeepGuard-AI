# DeepGuard AI 🛡️
![DeepGuard AI](https://img.shields.io/badge/Status-Active-success) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688) ![Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-f1e05a)

DeepGuard AI is a state-of-the-art Deepfake Face Detection engine powered by **EfficientNet-B4**. It provides real-time inference on face images to determine authenticity, along with GradCAM functionality to visualize which regions of the face manipulated the AI's decision. 

This project features a fully modular PyTorch training pipeline, a lightning-fast FastAPI backend, and a dynamic, beautiful Drag-and-Drop aesthetic frontend.

---

## 🚀 Features
- **EfficientNet-B4 Backbone**: High-accuracy Transfer Learning architecture.
- **GradCAM Explainability**: See *why* the model predicted an image as a Deepfake.
- **FastAPI Inference Server**: Robust asynchronous API endpoints with caching and typing.
- **Google Colab Ready**: Comes with an auto-generated all-in-one Jupyter Notebook for training heavily parameterized models directly on Cloud GPUs.

---

## 📁 Repository Structure
```
DeepGuard-AI/
├── app.py                 # FastAPI server (Backend)
├── frontend.html          # Web dashboard (Frontend UI)
├── predict.py             # CLI Inference & GradCAM generator
├── model.py               # DeepGuardNet PyTorch architecture (EfficientNet)
├── train.py               # Local training script 
├── dataset.py             # Data loader, augmentations & transformations
├── utils.py               # Utilities (Loss functions, seed matching, logging)
├── build_notebook.py      # Compiler script that generates DeepGuard_AI.ipynb
├── DeepGuard_AI.ipynb     # All-in-one Colab Notebook (Generated)
├── requirements.txt       # Python dependencies
└── best_model.pth         # Saved weights (Train on Colab and place here)
```

---

## ⚙️ Installation

1. **Clone the repository** (if applicable) and open the project directory.
2. **Install all dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🎯 How to Run (Inference)

### 1. Start the API Server
Ensure your `best_model.pth` file is located in the root directory. Spin up the FastAPI backend using Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
> **Note:** If you get an `[Errno 10048] address already in use` error, it signifies that the server is already running in the background! You can either close that background process or proceed to step 2.

### 2. Open the Web Dashboard
Right-click on `frontend.html` and open it in your preferred Desktop browser (Chrome, Edge, Safari). Alternatively, run this in a new terminal:
```bash
# Windows
Start-Process frontend.html

# Mac 
open frontend.html
```

### 3. Drag and Drop!
Drag any human or AI-generated face image directly onto the Upload zone and get a sub-second authenticity verdict!

---

##  Model Training

You have two options to train the model:

### Option A: The Cloud Way (Recommended)
Upload `DeepGuard_AI.ipynb` directly into **Google Colab**. Ensure you have a `kaggle.json` ready. The notebook automatically downloads the dataset, handles memory limits, mounts your Google Drive, and saves the trained `best_model.pth` directly to your cloud storage.

### Option B: The Local Modular Way
Download the dataset manually to your local machine, and run:
```bash
python train.py --data_dir /path/to/extracted/dataset --epochs 30 --batch_size 32
```

---

## 📸 CLI Predictions & Heatmaps
Don't want to use the web server? You can query the predictor directly via Terminal and generate X-Ray (GradCAM) deepfake heatmaps:

```bash
python predict.py --image test_face.jpg --gradcam
```
*This will output the probability metrics in the console and save a `test_face_gradcam.png` visualization image locally!*

