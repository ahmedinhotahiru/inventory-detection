# Inventory Detection System

![Inventory Detection Banner](https://img.shields.io/badge/Inventory-Detection-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)

An automated system to detect and count Pepsi and Kilimanjaro products in shelf images, solving the time-consuming problem of manual inventory counting.

## 🌟 Features

- **Multi-Tier Implementation**: Three different approaches with increasing complexity
- **Product Recognition**: Identifies Pepsi and Kilimanjaro bottles in complex shelf images
- **Visual Results**: Generates annotated images with bounding boxes and counts
- **Detailed Reporting**: Outputs product counts in structured JSON format
- **Mobile Compatibility**: Optimized for deployment to Android/iOS devices

## 📋 Problem Statement

Manual inventory counting:
- Takes hours each week
- Is often skipped due to time constraints
- Leads to inventory errors and stock management issues

## 🔍 Solution

This project provides three progressively sophisticated solutions:

### 1. API Approach (Gemini)
Utilizes Google's Gemini multimodal API to quickly identify products without the need for specialized hardware or training.

### 2. GPU Approach (YOLOv8)
Implements YOLOv8 with ResNet50 feature extraction for higher precision detection, leveraging GPU acceleration.

### 3. Mobile Approach (TensorFlow Lite)
Creates a lightweight model that can run directly on mobile devices for field use, even in areas with poor connectivity.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Jupyter Notebooks**: Interactive development and demonstration
- **TensorFlow/Keras**: For mobile model development
- **PyTorch/YOLOv8**: For GPU-accelerated detection
- **Google Gemini API**: For cloud-based detection
- **OpenCV**: Image processing and visualization
- **Matplotlib**: Results visualization
- **NumPy/Pandas**: Data manipulation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Google Cloud and Google AI Studio accounts (for API approach)
- CUDA-capable GPU (for GPU approach)
- Android Studio (for mobile approach, optional)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/inventory-detection.git
cd inventory-detection
```

2. Create a virtual environment:
```bash
python -m venv inventory_env
source inventory_env/bin/activate  # On Windows: inventory_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📓 Jupyter Notebooks

For detailed exploration and explanation of each approach, see these notebooks:

- [`01_API_Approach.ipynb`](notebooks/1_API_Approach.ipynb): Gemini API implementation
- [`02_GPU_Approach.ipynb`](notebooks/2_GPU_Approach.ipynb): YOLOv8 implementation
- [`03_Mobile_Approach.ipynb`](notebooks/3_Mobile_Approach.ipynb): TensorFlow Lite implementation

## 📁 Project Structure

```
inventory_detection/
├── notebooks/                   # Jupyter notebooks implementation
│   ├── 1_API_Approach.ipynb     # Gemini API implementation
│   ├── 2_GPU_Approach.ipynb     # YOLOv8 implementation
│   └── 3_Mobile_Approach.ipynb  # TensorFlow Lite implementation
├── data/
│   ├── sample_images/           # Sample product images
│   └── inventory_images/        # Shelf photos to process
├── outputs/                     # Detection results
│   ├── api/
│   ├── gpu/
│   └── mobile/
├── voice_records/              # Voice explanations
│   ├── approaches.m4a
│   └── results.m4a
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎤 Voice Records

Two voice recordings are provided:
1. [`approaches.m4a`](voice_records/approaches.m4a): Methodology explanation before implementation
2. [`results.m4a`](voice_records/results.m4a): Findings explanation after implementation

## 🔄 Usage Instructions

### API Approach (Gemini)

1. Get an API key:
   - Go to [`Google AI Studio`](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Save it in a `.env` file:
   ````bash
   GOOGLE_API_KEY=your_api_key_here
   ```

2. Run the API approach notebook:
```bash
jupyter notebook notebooks/01_API_Approach.ipynb
```

### GPU Approach (YOLOv8)

1. Install the required dependencies:
```bash
pip install ultralytics torch torchvision scikit-learn opencv-python
```

2. Run the GPU approach notebook:
```bash
jupyter notebook notebooks/02_GPU_Approach.ipynb
```

### Mobile Approach (TensorFlow Lite)

1. Install the required dependencies:
```bash
pip install tensorflow tensorflow-hub numpy pillow matplotlib
```

2. Run the Mobile approach notebook:
```bash
jupyter notebook notebooks/03_Mobile_Approach.ipynb
```

## 🔜 Future Improvements

1. **Instance Segmentation**: Implement for better handling of overlapping products
2. **Multi-Product Support**: Extend to recognize additional product types
3. **Real-time Processing**: Optimize for video stream processing
4. **Cloud Integration**: Connect with inventory management systems
5. **Automated Retraining**: Add capability to improve models with new data

## 🙏 Acknowledgments

- This project was created as an assignment for Dftr
- Sample images provided by Dftr

