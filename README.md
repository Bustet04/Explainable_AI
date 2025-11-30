# Explainable AI - Sleep Phase Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An academic project implementing **supervised** and **unsupervised** machine learning approaches for sleep phase detection using EEG/EOG/EMG physiological signals, with comprehensive **Explainable AI (XAI)** analysis.

---

## ⚡ Quick Start Guide (New Users)

**Want to try it immediately?** → Run the mental disorder notebooks (data included!)

**Want sleep analysis?** → Download Sleep-EDF dataset first (see instructions below)

### What's Included in This Repository

✅ **5 Training Notebooks** - Ready to run  
✅ **Mental Disorder EEG Dataset** - Included (no download needed)  
✅ **Source Code Modules** - Reusable preprocessing and feature extraction  
✅ **Results & Visualizations** - Example outputs from trained models  
✅ **Complete Documentation** - This README + inline notebook explanations  

❌ **Sleep-EDF Dataset** - NOT included (too large, ~2GB)  
❌ **Trained Models** - NOT included (you'll generate these locally)  

### Installation (3 steps, ~2 minutes)

```bash
# 1. Clone repository
git clone https://github.com/Bustet04/Explainable_AI.git
cd Explainable_AI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# OR: source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

**That's it!** Now jump to [Quick Start](#quick-start) below to choose which notebooks to run.

---

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
| **Notebook** | `notebooks/sleep_analysis/train_sleep_classifier.ipynb` | `notebooks/sleep_analysis/train_unsupervised_sleep_clustering.ipynb` |
| **Algorithm** | MLP Neural Network | Autoencoder + K-Means |
| **Labels Required** | Yes (for training) | No (discovery-based) |
| **Performance** | 87% test accuracy | ARI: 0.65, NMI: 0.62 |
| **Interpretability** | Medium (feature importance) | High (4 XAI components) |
| **Training Time** | ~2-5 minutes | ~10-15 minutes (includes XAI) |
| **Use Case** | Clinical deployment, automation | Pattern discovery, research, exploration |
| **Advantage** | Precise predictions | No labeled data needed, explainable patterns |
| **Data Required** | Sleep-EDF dataset (download) | Sleep-EDF dataset (download) |

## 📁 Project Structure

## 📁 Project Structure

```
Explainable_AI/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── GIT_WORKFLOW.md                   # Git usage guide
│
├── notebooks/                         # Training notebooks
│   ├── sleep_analysis/
│   │   ├── train_sleep_classifier.ipynb           # Supervised MLP (87% accuracy)
│   │   └── train_unsupervised_sleep_clustering.ipynb  # Unsupervised + XAI
│   └── mental_disorders/
│       ├── train_binary_randomforest.ipynb        # Random Forest classifier
│       ├── train_engineered_features.ipynb        # Feature engineering approach
│       └── train_neural_network.ipynb             # Deep learning classifier
│
├── src/                               # Modular source code
│   ├── preprocessing.py               # Signal preprocessing utilities
│   ├── features.py                    # Feature extraction functions
│   ├── model.py                       # Model architectures
│   ├── data_preprocessing_v2.py       # Extended preprocessing pipeline
│   ├── data_summary_v2.py             # Dataset statistics
│   └── ui_streamlit.py                # Interactive demo app
│
├── data/                              # Data directory
│   ├── mental_disorders/
│   │   └── EEG.machinelearing_data_BRMH.csv  # Mental disorder EEG dataset (included)
│   ├── raw/                           # Raw sleep signals (download separately)
│   └── processed/                     # Preprocessed features (generated by notebooks)
│
├── models/                            # Trained models (generated locally, excluded from repo)
│   ├── saved_models/                  # Supervised classifiers
│   └── unsupervised_clustering/       # Autoencoder, K-Means, XAI results
│
├── results/                           # Output artifacts
│   ├── visualizations/                # Training plots, confusion matrices
│   └── explainability/                # XAI analysis results
│
└── docs/                              # Documentation
    └── report.md                      # Analysis report template
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

#### **Option 1: Mental Disorder Classification** (Easiest - data included!)

The mental disorder classification notebooks come with data included, so you can start immediately:

```bash
# Start Jupyter
jupyter notebook

# Open any of these notebooks:
# - notebooks/mental_disorders/train_binary_randomforest.ipynb
# - notebooks/mental_disorders/train_engineered_features.ipynb
# - notebooks/mental_disorders/train_neural_network.ipynb
```

**What you'll get:**
- Pre-loaded EEG dataset for mental disorder classification
- Multiple ML approaches (Random Forest, feature engineering, neural networks)
- Ready to run without additional data downloads
- Training time: ~2-5 minutes

#### **Option 2: Sleep Analysis** (Requires dataset download)

For sleep stage classification, you need to download the Sleep-EDF dataset first:

**Step 1: Download Sleep-EDF Dataset**

```bash
# Download from PhysioNet (option 1 - manual)
# Visit: https://physionet.org/content/sleep-edfx/1.0.0/
# Download the sleep-cassette files to data/raw/

# OR use wget (option 2 - command line)
cd data/raw
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

**Step 2: Run Preprocessing (First Time Only)**

```python
# Run preprocessing to generate features
from src.data_preprocessing_v2 import SleepDataPreprocessor

preprocessor = SleepDataPreprocessor('data/raw/sleep-cassette')
features, labels = preprocessor.process_all_recordings()

# Save processed data
import numpy as np
np.save('data/processed/features.npy', features)
np.save('data/processed/labels.npy', labels)
```

**Step 3: Train Models**

```bash
# Supervised classifier (87% accuracy)
jupyter notebook notebooks/sleep_analysis/train_sleep_classifier.ipynb

# Unsupervised clustering with XAI
jupyter notebook notebooks/sleep_analysis/train_unsupervised_sleep_clustering.ipynb
```

**What you'll get:**
- 87% classification accuracy on supervised model
- Comprehensive XAI analysis on unsupervised approach
- 20+ visualizations in results/
- Trained models saved in models/
- Training time: 10-15 minutes total

## 📊 Dataset

### Mental Disorder Classification Dataset (Included ✅)

**File**: `data/mental_disorders/EEG.machinelearing_data_BRMH.csv`

This dataset is included in the repository and ready to use immediately.

**What it contains:**
- EEG features for mental disorder classification
- Pre-processed and ready for training
- Used by notebooks in `notebooks/mental_disorders/`

### Sleep-EDF Database (Download Required 📥)

**Source**: [Sleep-EDF Database Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/)

⚠️ **Note**: Due to size constraints, the Sleep-EDF dataset is NOT included in the repository. You must download it separately.

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

**Download Instructions:**

1. **Manual Download:**
   - Visit: https://physionet.org/content/sleep-edfx/1.0.0/
   - Download sleep-cassette folder
   - Extract to `data/raw/sleep-cassette/`

2. **Command Line (Linux/macOS):**
   ```bash
   cd data/raw
   wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
   ```

3. **After Download:**
   - Run preprocessing script (see Quick Start above)
   - Generated files: `data/processed/features.npy` and `labels.npy`
   - These are excluded from git (too large) but notebooks will generate them

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

After running the notebooks, the following artifacts are automatically generated:

**Supervised Classifier** (`models/saved_models/`):
- `sleep_classifier.joblib` - Trained MLP model
- `pytorch_sleep_model.pt` - PyTorch model weights
- `feature_scaler.joblib` - StandardScaler for features

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

**Results & Visualizations** (`results/`):
- `results/visualizations/` - Training curves, confusion matrices, feature distributions
- `results/explainability/` - XAI analysis outputs, feature importances, decision trees

⚠️ **Note**: Models are excluded from git (`.gitignore`) due to size. They are generated when you run the notebooks locally.

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

## 🔧 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'X'"**
```bash
# Make sure virtual environment is activated
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/macOS

# Reinstall dependencies
pip install -r requirements.txt
```

**2. "FileNotFoundError: data/processed/features.npy"**
- For sleep analysis: You need to download and preprocess the Sleep-EDF dataset first
- See "Quick Start - Option 2" above for download instructions
- For mental disorders: Data is included, check path `data/mental_disorders/EEG.machinelearing_data_BRMH.csv`

**3. "Kernel died / Out of memory"**
- Sleep dataset is large (~415K samples)
- Requires minimum 8GB RAM
- Try closing other applications
- Or reduce batch size in training notebooks

**4. "Results/models folders are empty"**
- These are generated when you run notebooks
- Model files are excluded from git (`.gitignore`)
- Run notebooks to generate them locally

**5. "Notebook kernel not found"**
```bash
# Install ipykernel in virtual environment
pip install ipykernel
python -m ipykernel install --user --name=explainable_ai
```

### Where to Get Help

- 📖 Check `GIT_WORKFLOW.md` for git-specific issues
- 📊 Review `results/` for example outputs
- 📝 Read notebook markdown cells for detailed explanations
- 🐛 Open an issue on GitHub for bugs  

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


