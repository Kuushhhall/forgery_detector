# 🔍 Image Forgery Detector

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange)](https://streamlit.io)

A professional forensic analysis tool to detect image manipulation and AI-generated content using multiple detection techniques.

## 🚀 Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your system

### One-Command Setup
```bash
# Clone this repository
git clone https://github.com/yourusername/forgery-detector.git
cd forgery-detector

# Build and run the container
docker build -t forgery-detector .
docker run -p 8501:8501 forgery-detector
```

Then open your browser to: **http://localhost:8501**

## 🎯 What It Does

This tool combines multiple forensic techniques to analyze images:

| Technique | What It Detects | Weight |
|-----------|-----------------|--------|
| **EXIF Metadata** | Missing camera info, suspicious software | 5% |
| **Error Level Analysis** | Compression inconsistencies | 15% |
| **Frequency Analysis** | Unnatural frequency patterns | 15% |
| **Noise Residual** | Camera sensor noise inconsistencies | 10% |
| **Copy-Move Detection** | Duplicated regions | 5% |
| **AI Detection (CLIP)** | AI-generated content | 30% |
| **CNN Classifier** | Custom deep learning models | 20% |

## 📸 How to Use

1. **Upload an image** (JPG, PNG, WebP supported)
2. **Wait for analysis** (15-30 seconds)
3. **Review results** with color-coded confidence scores:
   - 🟢 **0.0-0.35**: Likely authentic
   - 🟡 **0.35-0.6**: Suspicious, needs review  
   - 🔴 **0.6-1.0**: Likely forgery/AI
4. **Download detailed report** in JSON format

## 🎮 Interactive Features

### Adjust Detection Weights
- Modify importance of each detection technique
- Real-time score updates
- Customize for specific use cases

### Visual Analysis
- Error Level Analysis (ELA) overlays
- Frequency spectrum visualization
- Copy-move region highlighting
- Noise pattern analysis

## 🔧 Advanced Usage

### GPU Acceleration
```bash
docker run --gpus all -p 8501:8501 forgery-detector
```

### Custom Configuration
```bash
# Different port
docker run -p 8080:8501 forgery-detector

# Background mode
docker run -d -p 8501:8501 --name forgery-detector forgery-detector

# Resource limits
docker run -p 8501:8501 --memory=4g --cpus=2 forgery-detector
```

### Development Mode
```bash
# Run without Docker
pip install -r requirements.txt
streamlit run app.py
```

## 🏗️ Architecture

```
forgery-detector/
├── app.py              # Main Streamlit application
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
├── models/
│   └── clip_probe.py   # CLIP-based AI detection
└── README.md          # This file
```

## 📊 Detection Techniques

### EXIF Metadata Analysis
- Checks for missing or suspicious metadata
- Flags images from editing software
- Validates camera and timestamp information

### Error Level Analysis (ELA)
- Compares original vs recompressed image
- Highlights areas with different compression levels
- Detects copy-paste and editing artifacts

### Frequency Domain Analysis
- Uses Fast Fourier Transform (FFT)
- Identifies unnatural frequency patterns
- Detects AI-generated image signatures

### Noise Residual Analysis
- Extracts camera sensor noise (PRNU proxy)
- Analyzes noise uniformity across image
- Detects inconsistencies in sensor patterns

### Copy-Move Detection
- Uses ORB feature matching
- Finds duplicated regions
- Highlights cloned objects

### AI Content Detection
- Uses OpenAI CLIP model
- Semantic analysis of image content
- Distinguishes real vs synthetic images

## ⚠️ Important Limitations

### What It CAN Do:
- Provide probabilistic confidence scores
- Identify common manipulation patterns
- Detect AI-generated content
- Give detailed technical analysis

### What It CANNOT Do:
- Provide definitive proof (always probabilistic)
- Detect all types of forgeries
- Replace human expert analysis
- Be used as legal evidence alone

## 🆘 Troubleshooting

### Common Issues

**Port already in use:**
```bash
docker run -p 8502:8501 forgery-detector
```

**Container won't start:**
```bash
docker logs forgery-detector
```

**Out of memory:**
```bash
docker run -p 8501:8501 --memory=4g forgery-detector
```

**First run is slow:**
- Models download on first use
- Subsequent runs are faster
- Models are cached in the container

### Performance Tips
- Use GPU if available for faster CLIP inference
- Allocate at least 4GB RAM for optimal performance
- First analysis may take longer due to model loading

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/forgery-detector/issues)
- **Questions**: Open a discussion
- **Email**: your.email@example.com

---

**Built with ❤️ for the forensic analysis community**

*Remember: This tool is for educational and preliminary analysis. Always verify critical findings with human experts.*
