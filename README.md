# Vehicle Detection Binary Classification

A deep learning project for binary vehicle detection using EfficientNet-B0 architecture, built with PyTorch and deployed via FastAPI.

## 🚗 Project Overview

This project implements a binary image classification system to detect the presence of vehicles in images. The model is built using EfficientNet-B0 as the backbone architecture and provides a REST API for real-time inference with a web interface.

## 🏗️ Architecture

- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Framework**: PyTorch
- **API**: FastAPI
- **Task**: Binary Classification (Vehicle/No Vehicle)
- **Interface**: Web-based UI with static HTML

## 📋 Features

- High-accuracy vehicle detection using state-of-the-art EfficientNet architecture
- Fast inference optimized for real-time applications
- RESTful API with automatic documentation
- Web interface for easy image upload and testing
- Pre-trained model checkpoints ready for deployment
- Image preprocessing pipeline with data augmentation

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rockstabber1/vehicle-detection.git
cd vehicle-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv3
source venv3/bin/activate  # On Windows: venv3\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
app/
├── __pycache__/          # Python cache files
├── main.py               # FastAPI application entry point
├── model_utils.py        # Model utilities and helper functions
├── Dataset/              # Training and validation datasets
├── Models/               # Model-related files
│   ├── .ipynb_checkpoints/
│   ├── finalmodel1.pkl   # Trained model checkpoint
│   └── tyremodel.pkl     # Alternative model checkpoint
├── raw_data/             # Raw dataset files
├── static/               # Static files for web interface
│   ├── index.html        # Web interface HTML
│   └── testfolder/       # Test images
└── venv3/                # Virtual environment
```

## 🚀 Quick Start

### Running the Application

1. Ensure your dataset is in the `Dataset/` directory
2. Make sure your trained model (`finalmodel1.pkl`) is in the `Models/` directory
3. Navigate to the app directory:
```bash
cd app
```

4. Start the FastAPI server:
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5. Access the web interface at `http://localhost:8000` or API documentation at `http://localhost:8000/docs`

### Docker Deployment

```bash
docker-compose up --build
```

## 📊 API Endpoints

### Web Interface

**GET** `/`

Access the web interface for uploading images and getting predictions.

### Predict Vehicle

**POST** `/predict`

Upload an image to get vehicle detection prediction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "prediction": "vehicle",
  "confidence": 0.95,
  "processing_time": 0.123
}
```

### Health Check

**GET** `/health`

Check API status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🧪 Usage Examples

### Web Interface

1. Open your browser and go to `http://localhost:8000`
2. Upload an image using the web form
3. View the prediction results instantly

### Python Client

```python
import requests

# Upload image for prediction
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

## 📈 Model Performance

| Metric    | Value |
|-----------|-------|
| Accuracy  | 94.2% |
| Precision | 93.8% |
| Recall    | 94.6% |
| F1-Score  | 94.2% |

*Note: Update these metrics based on your actual model performance*

## 🔧 Configuration

Key parameters can be configured in your Python files:

```python
# Model parameters
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 2
INPUT_SIZE = 224
MODEL_PATH = "Models/finalmodel1.pkl"

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 8000
```

## 📝 File Descriptions

- **`main.py`**: FastAPI application with endpoints and web interface serving
- **`model_utils.py`**: Utility functions for model loading, preprocessing, and inference
- **`static/index.html`**: Web interface for image upload and prediction display
- **`Models/finalmodel1.pkl`**: Primary trained model checkpoint
- **`Models/tyremodel.pkl`**: Alternative model (possibly for tire detection)
- **`Dataset/`**: Contains training and validation data
- **`raw_data/`**: Raw, unprocessed dataset files

## 🐳 Docker

### Build Image

```bash
docker build -t vehicle-detection .
```

### Run Container

```bash
docker run -p 8000:8000 vehicle-detection
```

### Dockerfile Example

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 📝 Requirements

```
torch>=1.12.0
torchvision>=0.13.0
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
Pillow>=8.3.0
numpy>=1.21.0
opencv-python>=4.5.0
efficientnet-pytorch>=0.7.1
pydantic>=1.8.0
jinja2>=3.0.0
aiofiles>=0.7.0
```

## 🛠️ Development

### Model Training

To train your own model, ensure your dataset is properly structured in the `Dataset/` directory and use your training scripts. The trained model should be saved as a `.pkl` file in the `Models/` directory.

### Adding New Features

1. Update `model_utils.py` for new model functionalities
2. Modify `main.py` to add new API endpoints
3. Update `static/index.html` for new web interface features

## 🐛 Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `finalmodel1.pkl` exists in `Models/` directory
2. **Import errors**: Check if all dependencies are installed in your virtual environment
3. **Port conflicts**: Change port in main.py if 8000 is occupied
4. **Memory issues**: Consider using CPU inference if GPU memory is insufficient

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 👥 Authors

- Your Name - [@rockstabber1](https://github.com/rockstabber1)

## 🙏 Acknowledgments
- **Dataset Source**: [Vehicle Detection Image Set by brsdincer on Kaggle](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set)
- EfficientNet paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- PyTorch team for the excellent deep learning framework
- FastAPI for the modern, fast web framework

---

⭐ If you found this project helpful, please give it a star!