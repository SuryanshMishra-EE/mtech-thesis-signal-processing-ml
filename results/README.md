# EEG Preprocessing Results

This folder contains quantitative and visual validation results of the EEG preprocessing pipeline applied prior to feature extraction and machine learning.

## Objectives
The preprocessing validation aims to verify:
- Signal quality consistency across subjects and load conditions
- Effective removal of motion and muscle artifacts
- Channel integrity after bad-channel detection and interpolation
- Spectral consistency of EEG signals across subjects

## Summary of Validation Analyses

### 1. Signal Quality and Artifact Rejection
- All retained epochs exhibit peak-to-peak amplitudes below the ±300 µV rejection threshold.
- Baseline drift across channels remains within ±5 µV, confirming effective DC offset correction and re-referencing.

### 2. Signal-to-Noise Ratio (SNR)
- SNR values remain consistently above 2 across all subjects and load conditions.
- Motor cortex channels (C3, Cz) show higher SNR compared to peripheral channels.

### 3. Channel Interpolation
- Fewer than 8% of channels required interpolation across the full dataset.
- Interpolation rates are consistent across subjects, indicating stable electrode contact and acquisition quality.

### 4. Spectral Validation
- Power spectral density (PSD) profiles follow the expected 1/f EEG distribution.
- A consistent 50 Hz line-noise notch is observed, confirming effective notch filtering.

## Folder Structure
- `figures/` — Visual QC plots (PSD, SNR, baseline drift, interpolation)
- `tables/` — Numerical summaries and statistics used in thesis tables

