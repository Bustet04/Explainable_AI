# Models Directory

This directory stores trained model weights and serialized models.

## Subdirectories

### saved_models/
Supervised classification models:
- MLP classifier (87% accuracy)
- PyTorch models (.pt, .pth files)
- Scikit-learn models (.joblib files)

### unsupervised_clustering/
Unsupervised learning models:
- Autoencoder weights
- K-Means clustering models
- Explainability artifacts

**Note**: Model files are excluded from version control via `.gitignore` due to size
