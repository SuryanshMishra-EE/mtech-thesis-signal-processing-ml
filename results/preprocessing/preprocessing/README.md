## EEG Preprocessing Results

This folder contains quantitative and visual validation results of the EEG preprocessing pipeline applied before feature extraction and machine learning.

The objective of these analyses is to verify signal quality, artifact removal effectiveness, channel integrity, and spectral consistency across subjects and task conditions.

---

## Folder Structure

- `figures/`  
  Contains plots and visual summaries generated during preprocessing validation.

- `tables/`  
  Contains numerical summaries and quality-control statistics exported as tables.

---

## Preprocessing Pipeline

The preprocessing pipeline consists of the following steps:

1. Bandpass filtering (0.5–100 Hz)
2. Notch filtering at 50 Hz
3. Bad channel detection and removal
4. Channel interpolation
5. Artifact rejection (epoch-level)
6. Independent Component Analysis (ICA)
7. Re-referencing
8. Epoching around task markers

---

## Validation Metrics Included

The preprocessing quality is validated using:

- Epoch peak-to-peak amplitude thresholds (≤ 300 µV)
- Channel-wise baseline mean and drift
- Signal-to-noise ratio (SNR) per channel
- Subject-wise spectral power density (PSD)
- Channel interpolation statistics

These validations ensure that the EEG signals used for feature extraction and machine learning are physiologically meaningful and free from major artifacts.
