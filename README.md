# Explainable AI - Sleep Phase Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An academic project implementing **supervised** and **unsupervised** machine learning approaches for sleep phase detection using EEG/EOG/EMG physiological signals, with comprehensive **Explainable AI (XAI)** analysis.

## 🎯 Project Overview

This repository contains two complementary approaches to automatic sleep stage classification:

### 1. **Supervised Sleep Classification** (87% Accuracy)
A Multi-Layer Perceptron (MLP) neural network trained on labeled sleep data to predict 5 sleep stages (Wake, REM, N1, N2, N3).

**Key Features:**
- Binary classification architecture with softmax output
- Spectral feature engineering (EEG frequency bands: delta, theta, alpha, beta, gamma)
- Statistical features from EOG and EMG signals
- Comprehensive evaluation metrics and confusion matrices
- Real-world clinical deployment potential

### 2. **Unsupervised Sleep Pattern Discovery** (Clustering + XAI)
An Autoencoder-based approach to discover sleep patterns without using labels, validated with 4 XAI components.

**Key Features:**
- Deep autoencoder for latent representation learning (24D → 8D compression)
- PCA dimensionality reduction for visualization
- K-Means clustering to discover 5 distinct sleep phases
- **4 Comprehensive XAI Components:**
  1. **Latent Space Explainability**: PCA visualization, feature correlations, reconstruction quality
  2. **Cluster Explainability**: Prototypes, EEG band power analysis, physiological interpretation
  3. **Feature Attribution**: Random Forest surrogate model, permutation importance
  4. **Stability Analysis**: Multi-run validation, reproducibility metrics

### Comparison: Supervised vs Unsupervised

| Aspect | Supervised Classifier | Unsupervised Clustering |
|--------|----------------------|-------------------------|
| **Notebook** | `train_sleep_classifier.ipynb` | `train_unsupervised_sleep_clustering.ipynb` |
| **Algorithm** | MLP Neural Network | Autoencoder + K-Means |
| **Labels Required** | Yes (for training) | No (discovery-based) |
| **Performance** | 87% test accuracy | ARI: 0.65, NMI: 0.62 |
| **Interpretability** | Medium (feature importance) | High (4 XAI components) |
| **Training Time** | ~2-5 minutes | ~10-15 minutes (includes XAI) |
| **Use Case** | Clinical deployment, automation | Pattern discovery, research, exploration |
| **Advantage** | Precise predictions | No labeled data needed, explainable patterns |

## 📁 Project Structure

## 📁 Project Structure

```
Explainable_AI/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── GIT_WORKFLOW.md                   # Git usage guide
│
├── Code/Projekt/                      # Main project notebooks (original location)
│   ├── train_sleep_classifier.ipynb           # Supervised MLP (87% accuracy)
│   ├── train_unsupervised_sleep_clustering.ipynb  # Unsupervised + XAI
│   ├── data_preprocessing.py          # Feature extraction pipeline
│   ├── data_summary.py                # Dataset statistics
│   └── ui_streamlit.py                # Interactive demo app
│
├── data/
│   ├── raw/                          # Raw physiological signals (excluded)
│   └── processed/                    # Preprocessed features (excluded)
│       ├── features.npy              # 24 engineered features per epoch
│       └── labels.npy                # Sleep stage labels (N1, N2, N3, REM, Wake)
│
├── src/                              # Modular source code
│   ├── preprocessing.py              # Signal preprocessing utilities
│   ├── features.py                   # Feature extraction functions
│   └── model.py                      # Model architectures
│
├── notebooks/                        # Jupyter notebooks (clean versions for repo)
│   └── [Cleaned notebooks will be moved here]
│
├── docs/                             # Documentation
│   └── report.md                     # Analysis report template
│
├── models/                           # Trained models (excluded from repo)
│   ├── saved_models/                 # Supervised classifiers
│   └── unsupervised_clustering/      # Autoencoder, K-Means, XAI results
│
└── results/                          # Output artifacts (excluded from repo)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM (for processing ~415K samples)
- GPU optional (CPU works, training takes ~5-15 min)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bustet04/Explainable_AI.git
   cd Explainable_AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

#### **1. Supervised Sleep Classification** (Recommended for beginners)

```bash
jupyter notebook Code/Projekt/train_sleep_classifier.ipynb
```

**What you'll get:**
- 87% classification accuracy on test set
- Confusion matrix and per-class metrics
- Feature importance analysis
- Training time: ~2-5 minutes

#### **2. Unsupervised Pattern Discovery** (Advanced XAI)

```bash
jupyter notebook Code/Projekt/train_unsupervised_sleep_clustering.ipynb
```

**What you'll get:**
- Discovered 5 sleep phase clusters (unsupervised)
- 4 comprehensive XAI components
- 20+ publication-quality visualizations
- Complete analysis report
- Training time: ~10-15 minutes

## 📊 Dataset

**Source**: [Sleep-EDF Database Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/)

**Physiological Signals:**
- 2 EEG channels (Fpz-Cz, Pz-Oz)
- 1 EOG channel (horizontal eye movement)
- 1 EMG channel (submental chin muscle activity)

**Data Statistics:**
- **Total epochs**: ~415,000 (30-second windows)
- **Sleep stages**: 5 classes
  - Wake (W)
  - REM sleep (R)
  - Non-REM Stage 1 (N1) - light sleep
  - Non-REM Stage 2 (N2) - deeper sleep
  - Non-REM Stage 3 (N3) - deep sleep / slow-wave

**Engineered Features** (24 per epoch):
- **EEG features** (per channel): Delta, theta, alpha, beta, gamma band powers + mean, std, kurtosis
- **EOG features**: Mean, std, max amplitude, zero crossings
- **EMG features**: Mean, std, absolute mean, energy

## 🧠 Model Architectures

### 1. Supervised MLP Classifier

```
Input (24 features)
    ↓
Dense (64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense (32) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense (16) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Output (5 classes) + Softmax
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 50 with early stopping
- Performance: **87% test accuracy**

### 2. Unsupervised Autoencoder

```
Encoder:                      Decoder:
Input (24 features)           Latent (8D)
    ↓                             ↓
Dense (16) + ReLU            Dense (16) + ReLU
BatchNorm + Dropout(0.2)     BatchNorm + Dropout(0.2)
    ↓                             ↓
Dense (8) + ReLU             Dense (24)
BatchNorm                    
    ↓                         
[Latent Space: 8D]           Output (24 features)
```

**Training Configuration:**
- Compression: 24D → 8D (3x reduction)
- Loss: MSE (reconstruction error)
- Optimizer: Adam (lr=0.001) with weight decay
- Epochs: 50 with validation monitoring

**Post-Processing:**
- PCA: 8D → 3D for visualization
- K-Means: Discover K=5 clusters

## 🔍 Explainable AI (XAI) Components

The unsupervised approach includes **4 comprehensive XAI analyses**:

### 1. **Latent Space Explainability**
**Goal**: Understand what the autoencoder learned

- **2D/3D PCA Visualizations**: Cluster separation in latent space
- **PCA Loadings Analysis**: Which latent dimensions contribute to each principal component
- **Reconstruction Quality**: Per-cluster error distribution
- **Insight**: Verify that latent space captures meaningful sleep patterns

### 2. **Cluster Explainability**
**Goal**: Interpret what each cluster represents physiologically

- **Cluster Prototypes**: Mean spectral fingerprint per cluster
- **EEG Band Power Profiles**: Delta, theta, alpha, beta, gamma across clusters
- **Confusion Matrix**: Discovered clusters vs. true sleep stages
- **Purity Analysis**: Dominant sleep stage per cluster
- **Insight**: Map discovered clusters to physiological sleep stages (W, N1, N2, N3, REM)

### 3. **Feature Attribution**
**Goal**: Identify which features drive cluster assignments

- **Surrogate Random Forest**: Train RF to predict clusters from features
- **Gini Feature Importance**: Information gain from tree splits
- **Permutation Importance**: Accuracy drop when feature shuffled
- **Method Comparison**: Validate robust features across methods
- **Insight**: Delta/theta band powers are most discriminative

### 4. **Stability Analysis**
**Goal**: Verify clusters are reproducible, not random artifacts

- **Multi-Run Testing**: 10 clusterings with different random seeds
- **Adjusted Rand Index (ARI)**: Agreement between clusterings
- **Normalized Mutual Information (NMI)**: Information overlap
- **Pairwise Stability Matrix**: Run-to-run consistency heatmap
- **Insight**: High stability (ARI > 0.9) confirms well-defined patterns

## 📈 Results & Performance

### Supervised Classifier
- **Test Accuracy**: 87%
- **Per-Class Performance**:
  - Wake: ~92% F1-score
  - N2/N3: ~85-88% F1-score  
  - N1/REM: ~75-80% F1-score (harder to distinguish)
- **Training Time**: 2-5 minutes (CPU)

### Unsupervised Clustering
- **Discovered Clusters**: 5 (matching known sleep stages)
- **Reconstruction Error**: 0.01-0.05 MSE
- **Clustering Quality**: 
  - Silhouette Score: ~0.45
  - Inertia: ~350,000
- **Alignment with True Labels**:
  - Adjusted Rand Index (ARI): 0.60-0.70
  - Normalized Mutual Information (NMI): 0.62-0.68
- **Stability**: 95%+ consistency across runs (ARI > 0.9)
- **Training Time**: 10-15 minutes (including XAI)

### Key Findings

**Physiological Insights:**
- **Cluster 0** → N3 (deep sleep): High delta power, low EMG
- **Cluster 1** → N2 (sleep): Moderate delta/theta, sleep spindles
- **Cluster 2** → N1 (drowsy): Mixed alpha/theta transition
- **Cluster 3** → REM: Low EMG, active EOG, theta activity
- **Cluster 4** → Wake: High alpha/beta, high EMG

**Feature Importance:**
1. **Delta band power** (EEG) - strongest discriminator
2. **Theta band power** (EEG) - drowsiness indicator
3. **EMG energy** - muscle activity (wake vs. sleep)
4. **Alpha band power** - wakefulness marker
5. **EOG amplitude** - eye movement (REM detection)

## 🛠️ Technologies & Dependencies

**Core Libraries:**
- **Deep Learning**: PyTorch 2.0+
- **Machine Learning**: scikit-learn
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Signal Processing**: SciPy
- **Model Persistence**: joblib

**Development Tools:**
- **Notebooks**: Jupyter Lab / Jupyter Notebook
- **Version Control**: Git
- **Environment**: Python virtual environment (.venv)

See `requirements.txt` for complete dependency list.

## 📂 Saved Models & Artifacts

After running the notebooks, the following artifacts are saved:

**Supervised Classifier** (`models/saved_models/`):
- `sleep_classifier.pth` - Trained MLP model
- `feature_scaler.joblib` - StandardScaler for features
- `label_encoder.joblib` - Encoder for sleep stage labels

**Unsupervised Clustering** (`models/unsupervised_clustering/`):
- `autoencoder.pth` - Trained autoencoder weights
- `feature_scaler.joblib` - StandardScaler
- `pca_model.joblib` - Fitted PCA model
- `kmeans_model.joblib` - Fitted K-Means model
- `surrogate_rf.joblib` - Random Forest for explainability
- `latent_representations.npy` - Encoded latent features
- `cluster_labels.npy` - Predicted cluster assignments
- `xai_results.joblib` - All XAI metrics and results
- `analysis_report.txt` - Comprehensive text report

## 🎓 Educational Value

This project demonstrates:

✅ **End-to-End ML Pipeline**: Data preprocessing → Feature engineering → Model training → Evaluation  
✅ **Supervised vs Unsupervised**: Compare two approaches on same dataset  
✅ **Deep Learning**: Autoencoder architecture, PyTorch implementation  
✅ **Clustering**: K-Means, elbow method, silhouette analysis  
✅ **Comprehensive XAI**: 4 complementary explainability techniques  
✅ **Domain Knowledge**: Physiological signal processing, sleep science  
✅ **Best Practices**: Modular code, reproducibility, documentation  
✅ **Evaluation Metrics**: ARI, NMI, silhouette, confusion matrices  

## 📖 Key Concepts Covered

- **Feature Engineering**: Spectral analysis (FFT), band power extraction
- **Dimensionality Reduction**: PCA for visualization and compression
- **Representation Learning**: Autoencoder latent space
- **Unsupervised Learning**: Clustering without labels
- **Model Explainability**: Multiple XAI techniques (LIME-like, feature attribution, stability)
- **Validation**: Cross-validation, stability analysis, external validation
- **Scientific Visualization**: Publication-quality plots

## 📚 References & Resources

**Dataset:**
- Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A., & Oberye, J. J. (2000). Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering*, 47(9), 1185-1194.
- Sleep-EDF Database: [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

**Sleep Stage Classification:**
- Iber, C., et al. (2007). The AASM Manual for the Scoring of Sleep and Associated Events. *American Academy of Sleep Medicine*.
- Rechtschaffen, A., & Kales, A. (1968). A manual of standardized terminology, techniques and scoring system for sleep stages of human subjects.

**XAI Methods:**
- Molnar, C. (2020). *Interpretable Machine Learning*. [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

**Autoencoder & Clustering:**
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Müller, A. C., & Guido, S. (2016). *Introduction to Machine Learning with Python*. O'Reilly.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add SHAP/LIME for instance-level explanations
- [ ] Implement temporal sequence modeling (LSTM/Transformer)
- [ ] Cross-subject validation analysis
- [ ] Real-time sleep monitoring demo
- [ ] Web-based interactive dashboard (Streamlit/Dash)
- [ ] Additional datasets (e.g., Sleep Cassette dataset)

**How to contribute:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m 'Add improvement'`
4. Push to branch: `git push origin feature/improvement`
5. Open Pull Request

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 👥 Author

**Erich** - [Bustet04](https://github.com/Bustet04)

*DHBW - Semester 5 - Explainable AI Project*

## 🙏 Acknowledgments

- **DHBW** - For the Explainable AI course and academic support
- **PhysioNet** - For providing the Sleep-EDF database
- **PyTorch Community** - For excellent deep learning framework
- **scikit-learn** - For robust ML tools
- **Open Source Community** - For all the amazing libraries

---

**⭐ If you find this project helpful, please consider giving it a star!**

**📧 Questions or feedback?** Open an issue or reach out via GitHub.

---

*Last updated: November 2025*


