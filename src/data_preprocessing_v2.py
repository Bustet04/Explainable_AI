import mne
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

class SleepDataPreprocessor:
    def __init__(self, data_dir):
        """
        Initialize the preprocessor with the directory containing EDF files
        
        Args:
            data_dir (str): Path to the directory containing sleep-cassette data
        """
        self.data_dir = Path(data_dir)
        self.sampling_rate = 100  # Standard sampling rate for processing
        
    def load_single_recording(self, psg_file, hypno_file):
        """
        Load and preprocess a single PSG recording and its corresponding hypnogram
        
        Args:
            psg_file (str): Name of the PSG EDF file
            hypno_file (str): Name of the hypnogram EDF file
            
        Returns:
            tuple: (preprocessed_signals, hypnogram)
        """
        # Load PSG data
        raw_psg = mne.io.read_raw_edf(self.data_dir / psg_file, preload=True)
        
        # Load hypnogram as annotations
        annotations = mne.read_annotations(self.data_dir / hypno_file)
        
        # Get channel names
        print(f"\nAvailable channels: {raw_psg.ch_names}")
        
        # Extract relevant channels (EEG, EOG, EMG)
        eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']  # Specific EEG channels in the dataset
        eog_channels = ['EOG horizontal']  # Specific EOG channel
        emg_channels = ['EMG submental']  # Specific EMG channel
        
        print("\nSelected channels:")
        print("EEG channels:", eeg_channels)
        print("EOG channels:", eog_channels)
        print("EMG channels:", emg_channels)
        
        # Create a copy of the raw data with only the channels we want
        channels_to_pick = eeg_channels + eog_channels + emg_channels
        raw_psg_selected = raw_psg.copy().pick_channels(channels_to_pick)
        
        # Resample data to standard sampling rate
        raw_psg_selected.resample(self.sampling_rate)
        
        # Apply filters
        raw_psg_selected.filter(l_freq=0.3, h_freq=45)  # Basic bandpass filter
        
        # Get the data as numpy array
        data = raw_psg_selected.get_data()
        
        # Create dictionary with the processed signals
        processed_signals = {
            'eeg': data[:2],  # First two channels are EEG
            'eog': data[2:3],  # Third channel is EOG
            'emg': data[3:],   # Fourth channel is EMG
        }
        
        return processed_signals, annotations
    
    def segment_into_epochs(self, signals, epoch_length=30):
        """
        Segment the continuous data into epochs
        
        Args:
            signals (dict): Dictionary containing EEG, EOG, and EMG signals
            epoch_length (int): Length of each epoch in seconds
            
        Returns:
            dict: Dictionary containing segmented signals
        """
        samples_per_epoch = epoch_length * self.sampling_rate
        segmented_signals = {}
        
        for signal_type, data in signals.items():
            n_channels, n_samples = data.shape
            n_epochs = n_samples // samples_per_epoch
            
            # Reshape data into epochs
            epochs = np.array([
                data[:, i * samples_per_epoch:(i + 1) * samples_per_epoch]
                for i in range(n_epochs)
            ])
            
            segmented_signals[signal_type] = epochs
            
        return segmented_signals
    
    def map_annotations_to_epochs(self, annotations, n_epochs, epoch_length=30.0):
        """
        Map annotations to epochs and convert to simplified sleep stages
        """
        onsets = annotations.onset
        durations = annotations.duration
        descriptions = annotations.description
        
        # Initialize labels array
        epoch_labels = np.full(n_epochs, None, dtype=object)
        
        # For each epoch, find the corresponding annotation
        for i in range(n_epochs):
            epoch_start = i * epoch_length
            epoch_end = (i + 1) * epoch_length
            
            # Find annotation that covers this epoch
            for onset, duration, desc in zip(onsets, durations, descriptions):
                if onset <= epoch_start < (onset + duration):
                    epoch_labels[i] = desc
                    break
        
        # Convert R&K stages to simplified AASM stages
        stage_map = {
            'Sleep stage W': 'W',    # Wake
            'Sleep stage R': 'R',    # REM
            'Sleep stage 1': 'N1',   # Non-REM 1
            'Sleep stage 2': 'N2',   # Non-REM 2
            'Sleep stage 3': 'N3',   # Non-REM 3
            'Sleep stage 4': 'N3',   # Non-REM 3 (combine 3&4)
            'Sleep stage ?': None    # Unknown/artifacts
        }
        
        mapped_labels = np.array([stage_map.get(label, None) for label in epoch_labels])
        return mapped_labels

    def plot_signals(self, signals, duration=30):
        """
        Plot the first few seconds of each signal type
        
        Args:
            signals (dict): Dictionary containing the signals
            duration (int): Duration in seconds to plot
        """
        n_samples = duration * self.sampling_rate
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        for i, (signal_type, data) in enumerate(signals.items()):
            time = np.arange(n_samples) / self.sampling_rate
            axes[i].plot(time, data[0, :n_samples])
            axes[i].set_title(f'{signal_type} Signal')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / "Daten" / "sleep-edf-database-expanded-1.0.0" / "sleep-cassette"
    
    print(f"Looking for data in: {data_dir}")
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please make sure the data is in the correct location")
        exit(1)
        
    preprocessor = SleepDataPreprocessor(data_dir)
    
    # Load first recording
    psg_file = "SC4001E0-PSG.edf"
    hypno_file = "SC4001EC-Hypnogram.edf"
    
    # Verify files exist
    if not (data_dir / psg_file).exists():
        print(f"Error: PSG file not found at {data_dir / psg_file}")
        exit(1)
    if not (data_dir / hypno_file).exists():
        print(f"Error: Hypnogram file not found at {data_dir / hypno_file}")
        exit(1)
    
    try:
        # Load and preprocess the data
        signals, annotations = preprocessor.load_single_recording(psg_file, hypno_file)
        
        # Print information about the annotations
        print("\nHypnogram Annotations:")
        print(f"Number of annotations: {len(annotations)}")
        print("Annotation descriptions:", set(annotations.description))
        
        # Segment the data into epochs
        segmented_signals = preprocessor.segment_into_epochs(signals)
        
        # Plot the first few seconds of the signals
        preprocessor.plot_signals(signals)
        
        print("\nData preprocessing completed successfully!")
        print(f"Segmented data shapes:")
        for signal_type, data in segmented_signals.items():
            print(f"{signal_type}: {data.shape}")
            
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")