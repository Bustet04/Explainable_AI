# Processed Data Directory

This directory contains preprocessed features extracted from raw signals.

## Sleep Analysis Features
- `features.npy`: 24 engineered features per 30-second epoch
- `labels.npy`: Sleep stage labels (N1, N2, N3, REM, Wake)

## Feature Engineering
Features include:
- EEG band powers (delta, theta, alpha, beta, gamma)
- Statistical measures (mean, std, skewness, kurtosis)
- EOG features (eye movement indicators)
- EMG features (muscle activity)

**Note**: Processed .npy files are excluded from version control due to size
