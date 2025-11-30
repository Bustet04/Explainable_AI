import numpy as np
import collections
from pathlib import Path

# Load data
data_dir = Path('data/processed')
features = np.load(data_dir / 'features.npy', allow_pickle=True)
labels = np.load(data_dir / 'labels.npy', allow_pickle=True)
predictions = np.load(data_dir / 'features.predictions.npy', allow_pickle=True)

# Calculate statistics
label_counts = collections.Counter(labels)

print('='*70)
print('SLEEP STAGE CLASSIFICATION PROJECT - DATA SUMMARY')
print('='*70)

print('\n📊 DATASET INFORMATION:')
print(f'  Total samples: {len(features):,}')
print(f'  Features per sample: {features.shape[1]}')
print(f'  Data type: Sleep EEG epochs (30-second windows)')

print('\n😴 SLEEP STAGE DISTRIBUTION:')
sleep_stages = {
    'W': 'Wake',
    'N1': 'Non-REM Stage 1 (light sleep)',
    'N2': 'Non-REM Stage 2',
    'N3': 'Non-REM Stage 3 (deep sleep)',
    'R': 'REM (Rapid Eye Movement)'
}

for stage in ['W', 'N1', 'N2', 'N3', 'R']:
    count = label_counts.get(stage, 0)
    pct = (count/len(labels))*100
    desc = sleep_stages.get(stage, '')
    print(f'  {stage:4s}: {count:7,} samples ({pct:5.2f}%) - {desc}')

none_count = label_counts.get(None, 0)
if none_count > 0:
    pct = (none_count/len(labels))*100
    print(f'  None: {none_count:7,} samples ({pct:5.2f}%) - unlabeled/invalid data')

print('\n🤖 MODEL STATUS:')
try:
    import joblib
    model = joblib.load('models/saved_models/sleep_classifier.joblib')
    print(f'  ✅ Model trained: MLPClassifier')
    print(f'  Architecture: {model.hidden_layer_sizes}')
    print(f'  Training complete: Yes')
    print(f'  Predictions available: {len(predictions):,} samples')
    
    # Calculate accuracy
    valid_mask = labels != None
    valid_labels = labels[valid_mask]
    valid_preds = predictions[valid_mask]
    accuracy = np.mean(valid_labels == valid_preds) * 100
    print(f'  Overall accuracy: {accuracy:.2f}%')
    
except Exception as e:
    print(f'  ❌ Error loading model: {e}')

print('\n📁 DATA FILES:')
print(f'  features.npy: {features.shape} - EEG features')
print(f'  labels.npy: {labels.shape} - Sleep stage labels')
print(f'  features.predictions.npy: {predictions.shape} - Model predictions')
print(f'  sleep_classifier.joblib: Trained MLP model')

print('\n' + '='*70)
print('The model has been trained on this sleep stage data!')
print('='*70)
