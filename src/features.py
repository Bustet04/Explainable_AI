"""
Feature extraction module for sleep signals.

This module computes spectral and statistical features from EEG/EOG/EMG epochs:
- EEG band powers (delta, theta, alpha, beta, gamma)
- Statistical features (mean, std, kurtosis)
- EOG features (amplitude, zero crossings)
- EMG features (energy, absolute mean)
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List


def compute_band_power(epoch: np.ndarray, fs: int = 100, 
                       band: str = 'delta') -> float:
    """
    Compute power in specific frequency band using Welch's method.
    
    Args:
        epoch: Signal epoch
        fs: Sampling frequency
        band: Frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')
        
    Returns:
        Power in the specified band
    """
    band_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Welch's periodogram
    freqs, psd = signal.welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
    
    # Get band power
    low, high = band_ranges[band]
    idx = np.logical_and(freqs >= low, freqs <= high)
    band_power = np.trapz(psd[idx], freqs[idx])
    
    return band_power


def extract_eeg_features(eeg_epoch: np.ndarray, fs: int = 100) -> Dict[str, float]:
    """
    Extract all EEG features for one epoch.
    
    Args:
        eeg_epoch: EEG signal epoch
        fs: Sampling frequency
        
    Returns:
        Dictionary of feature name -> value
    """
    features = {}
    
    # Band powers
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        features[f'{band}'] = compute_band_power(eeg_epoch, fs, band)
    
    # Statistical features
    features['mean'] = np.mean(eeg_epoch)
    features['std'] = np.std(eeg_epoch)
    features['kurtosis'] = stats.kurtosis(eeg_epoch)
    
    return features


def extract_eog_features(eog_epoch: np.ndarray) -> Dict[str, float]:
    """
    Extract EOG features for one epoch.
    
    Args:
        eog_epoch: EOG signal epoch
        
    Returns:
        Dictionary of feature name -> value
    """
    features = {
        'mean': np.mean(eog_epoch),
        'std': np.std(eog_epoch),
        'max_amplitude': np.max(np.abs(eog_epoch)),
        'zero_crossings': np.sum(np.diff(np.sign(eog_epoch)) != 0)
    }
    return features


def extract_emg_features(emg_epoch: np.ndarray) -> Dict[str, float]:
    """
    Extract EMG features for one epoch.
    
    Args:
        emg_epoch: EMG signal epoch
        
    Returns:
        Dictionary of feature name -> value
    """
    features = {
        'mean': np.mean(emg_epoch),
        'std': np.std(emg_epoch),
        'abs_mean': np.mean(np.abs(emg_epoch)),
        'energy': np.sum(emg_epoch ** 2)
    }
    return features


def extract_all_features(eeg_ch1: np.ndarray, eeg_ch2: np.ndarray,
                        eog: np.ndarray, emg: np.ndarray,
                        fs: int = 100) -> np.ndarray:
    """
    Extract all 24 features from one 30-second epoch.
    
    Args:
        eeg_ch1: EEG channel 1 epoch
        eeg_ch2: EEG channel 2 epoch
        eog: EOG epoch
        emg: EMG epoch
        fs: Sampling frequency
        
    Returns:
        Feature vector of length 24
    """
    feature_vector = []
    
    # EEG channel 1 (8 features)
    eeg1_feats = extract_eeg_features(eeg_ch1, fs)
    feature_vector.extend(eeg1_feats.values())
    
    # EEG channel 2 (8 features)
    eeg2_feats = extract_eeg_features(eeg_ch2, fs)
    feature_vector.extend(eeg2_feats.values())
    
    # EOG (4 features)
    eog_feats = extract_eog_features(eog)
    feature_vector.extend(eog_feats.values())
    
    # EMG (4 features)
    emg_feats = extract_emg_features(emg)
    feature_vector.extend(emg_feats.values())
    
    return np.array(feature_vector)


if __name__ == "__main__":
    print("Feature extraction module - use functions in your analysis pipeline")
    print("Total features: 24 (EEG1: 8, EEG2: 8, EOG: 4, EMG: 4)")
