# 🎯 Few-Shot Learning with Prototypical Networks

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> A powerful implementation of **Prototypical Networks** for few-shot learning on multiple datasets including Omniglot, MNIST, Fashion-MNIST, and KMNIST.

---

## ✨ Project Overview

**Few-shot learning** is a machine learning paradigm where models learn to recognize new classes from very few examples (typically 1-10 images per class). This project implements **Prototypical Networks**, a metric learning approach that learns to classify images by computing distances to prototype representations of each class.

### 🚀 Key Features

- ✅ **Multi-Dataset Support**: Omniglot, MNIST, Fashion-MNIST, KMNIST
- 🧠 **ResNet18 Architecture**: Pretrained on ImageNet for better feature extraction
- 📊 **N-way K-shot Learning**: Flexible configuration (5-way, 5-shot in baseline)
- 🎓 **40,000 Training Episodes**: Extensively trained for optimal performance
- 📈 **High Accuracy Results**: Achieving 97%+ on Omniglot dataset
- 🔍 **Metric Learning**: Distance-based classification using learned embeddings
- ⚡ **AdamW Optimizer**: Modern optimization with weight decay

---

## 📊 Experimental Results

### 🏆 Accuracy Summary

| Dataset | Pre-Training | Post-Training (40k episodes) | Improvement |
|---------|:------------:|:---------------------------:|:-----------:|
| **Omniglot** | 86.96% | **97.64%** | ⬆️ +10.68% |
| **MNIST** | 84.44% | - | - |
| **Fashion-MNIST** | 62.14% | - | - |
| **KMNIST** | 50.02% | - | - |

### ⚙️ Experiment Configuration

```
┌─────────────────────────────────────┐
│    TRAINING CONFIGURATION           │
├─────────────────────────────────────┤
│ Architecture        │ ResNet18       │
│ Backbone           │ ImageNet (PT)  │
│ N-way              │ 5 classes      │
│ N-shot             │ 5 examples     │
│ N-query            │ 10 per class   │
│ Training Episodes  │ 40,000         │
│ Validation Tasks   │ 100            │
│ Optimizer          │ AdamW          │
│ Learning Rate      │ 1e-3           │
│ Loss Function      │ CrossEntropy   │
│ Total Parameters   │ 11,176,512     │
└─────────────────────────────────────┘
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU acceleration)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/Nakshatra1729yuvi/Few-Shot-Learning_Prototypical_Network.git
cd Few-Shot-Learning_Prototypical_Network

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Prototypical_Networks.ipynb
```

### Required Packages

```
torch>=1.9.0
torchvision>=0.10.0
numpy
scikit-learn
matplotlib
pandas
```

---

## 📖 Usage

### 🎯 Quick Start

```python
# Load the pretrained model
model = ResNet18()
model.load_state_dict(torch.load('prototypical_network.pth'))
model.eval()

# Prepare your support set (few examples) and query set
support_images = load_support_set()  # Shape: (n_way, n_shot, C, H, W)
query_images = load_query_set()      # Shape: (n_way, n_query, C, H, W)

# Compute prototypes and make predictions
with torch.no_grad():
    support_embeddings = model(support_images)
    query_embeddings = model(query_images)
    # Compute distances and classify
    predictions = classify(support_embeddings, query_embeddings)
```

### 📚 Training Your Own Model

Open `Prototypical_Networks.ipynb` and follow the notebook cells:

1. **Data Loading**: Automatically downloads and prepares datasets
2. **Model Initialization**: Creates ResNet18 backbone
3. **Training Loop**: Trains for 40,000 episodes
4. **Evaluation**: Tests on validation tasks
5. **Visualization**: Plots accuracy curves and embeddings

---

## 📁 Project Structure

```
.
├── Prototypical_Networks.ipynb    # Main notebook with full implementation
├── README.md                       # This file
├── LICENSE                         # MIT License
└── requirements.txt                # Dependencies (to be added)
```

---

## 🧪 Model Architecture

### Prototypical Networks Pipeline

```
┌─────────────────────────────────────────┐
│  Support Set & Query Set                │
│  (Few examples + Query images)          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   ResNet18 Encoder   │
        │  (Feature Extractor) │
        └──────────┬───────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   ┌──────────────┐  ┌──────────────┐
   │  Prototypes  │  │ Query Embeds │
   │ (Mean of     │  │ (Embeddings) │
   │  support)    │  └──────┬───────┘
   └──────┬───────┘         │
          └─────────┬───────┘
                    ▼
        ┌───────────────────────┐
        │  Distance Computation │
        │  (Euclidean or other) │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │   Softmax + CrossEnt  │
        │   Classification Loss │
        └───────────┬───────────┘
                    ▼
             ┌──────────────┐
             │  Predictions │
             └──────────────┘
```

---

## 📈 Performance Metrics

### Omniglot Dataset Performance
- **Initial Accuracy**: 86.96%
- **Final Accuracy**: 97.64%
- **Total Improvement**: +10.68 percentage points
- **Training Episodes**: 40,000
- **Convergence**: Smooth and stable

### Why Prototypical Networks?

✨ **Advantages:**
- Intuitive metric-learning approach
- Fast adaptation to new classes
- Efficient computation during inference
- Strong performance on few-shot tasks
- Works well with limited data

---

## 🤝 Contributing

Contributions are welcome! 🎉

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👥 Authors & Credits

### Primary Author
- **Nakshatra1729yuvi** - Core implementation and research
  - GitHub: [@Nakshatra1729yuvi](https://github.com/Nakshatra1729yuvi)

### Acknowledgments

- 🙏 Original Prototypical Networks paper: [Snell et al., 2017](https://arxiv.org/abs/1703.05175)
- 📚 ResNet implementation: torchvision
- 🔬 Datasets: Omniglot, MNIST, Fashion-MNIST, KMNIST
- 🏫 Research inspiration from the few-shot learning community

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Nakshatra1729yuvi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📞 Support & Issues

If you encounter any issues or have questions:

- 🐛 **Report bugs**: [Issues](https://github.com/Nakshatra1729yuvi/Few-Shot-Learning_Prototypical_Network/issues)
- 💬 **Discussions**: Open a GitHub Discussion
- 📧 **Contact**: Reach out via GitHub

---

## 🌟 Star History

If this project helped you, please give it a ⭐ on GitHub!

---

<div align="center">

### Made with ❤️ by [Nakshatra1729yuvi](https://github.com/Nakshatra1729yuvi)

**Happy Learning! Keep Exploring the World of Few-Shot Learning 🚀**

</div>
