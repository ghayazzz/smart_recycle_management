# 🗑️ Waste Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.0%2B-06B6D4)

A deep learning system that classifies waste materials into 6 categories (cardboard, glass, metal, paper, plastic, trash) with a web interface for easy interaction.

## ✨ Features

- **Dual-Model Architecture**: 
  - Basic model (Basic) for fast inference
  - Enhanced model (Class Weights) with class balancing for better accuracy
- **Web Interface**:
  - Clean UI built with Tailwind CSS
  - Interactive results with Alpine.js
  - Side-by-side model comparison
- **Performance Metrics**:
  - Precision and F1 scores for both models
  - Class-wise performance analysis

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/waste-classifier.git
   cd waste-classifier
   ```

2. Download pre-trained models:
   - [Model 1 (Basic)](https://github.com/ghayazzz/smart_recycle_management/blob/main/models/non_cw_best_model.pth) → Save to `models/non_cw_best_model.pth`
   - [Model 2 (Enhanced)](https://github.com/ghayazzz/smart_recycle_management/blob/main/models/cw_best_model.pth) → Save to `models/cw_best_model.pth`

## 🚀 Usage

### Running the Flask API:
```bash
python app.py
```

Access the web interface at: http://localhost:5000

### API Endpoints:
- `POST /predict` - Accepts image file and returns classification results
- `GET /` - Serves the web interface

## 📊 Model Performance

| Metric        | Model 1 (Basic) | Model 2 (Enhanced) |
|--------------|----------------:|-------------------:|
| Precision    | 0.9940          | 1.0000             |
| F1 Score     | 0.9938          | 1.0000             |
| Validation Accuracy | 0.9930   | 0.9938             |

**Class-wise F1 Scores:**
```python
{
  "cardboard": 0.9901,
  "glass": 0.9722,
  "metal": 0.9817,
  "paper": 0.9912,
  "plastic": 0.9765,
  "trash": 0.9429
}
```

## 🛠️ Project Structure

```
waste-classifier/
├── backend                 #Flask application
│    └── app.py
├── dataset/
│   ├── trashnet
│   └── trashnet.zip
├── models/
│   ├── cw_best_model.pth    #Basic model weights
│   └── non_cw_best_model.pth    #Enhanced model weights
├── templates/
│   └── index.html         #Web interface
└── README.md              #This file
```

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Project Maintainer - [Mohammed Ghayasuddin](mailto:mghayasuddin2000@gmail.com)

---

Made with ♥ by Mohammed Ghayasuddin
