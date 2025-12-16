# Experimental Results Summary

This directory contains the consolidated experimental results of the M.Tech thesis titled  
**“Signal Processing and Machine Learning for EEG-Based Task Weight Classification.”**

The results are organized into three stages corresponding to the complete pipeline:
1. EEG preprocessing validation  
2. Regression-based weight estimation  
3. Classification-based weight recognition  

Each subfolder contains detailed quantitative results, figures, and observations.

---

## 1. EEG Preprocessing Results

**Path:** `results/preprocessing/`

This stage validates the quality and reliability of EEG signals before feature extraction and model training.

### Key Objectives
- Verify signal quality consistency across subjects and load conditions  
- Ensure effective removal of motion, muscle, and eye-movement artifacts  
- Validate channel integrity after bad-channel detection and interpolation  
- Confirm spectral consistency across subjects  

### Key Findings
- All epochs maintained peak-to-peak amplitudes below **300 μV**
- Mean baseline drift remained within **±5 μV** across channels
- Signal-to-noise ratio (SNR) consistently **> 2** for motor cortex channels (C3, Cz)
- Overall interpolated channels across all subjects: **~7.7%**, indicating good electrode contact and data quality
- Power spectral density (PSD) trends remained consistent across subjects and weights

➡️ See `results/preprocessing/README.md` for detailed figures and tables.

---

## 2. Regression Results (Weight Estimation)

**Path:** `results/regression/`

This stage evaluates the ability to **estimate continuous task weight** from EEG features using regression models.

### Model
- **Multiple Linear Regression (MLR)**
- Leave-One-Subject-Out (LOSO) cross-validation

### Key Results
- Mean Absolute Error (MAE): **≈ 0.36 kg**
- Predicted weights closely followed target weights across trials
- Regression performance indicates EEG features preserve continuous load-related information

### Interpretation
- The regression model demonstrates feasibility of **continuous cognitive load estimation**
- Errors are within acceptable limits given inter-subject EEG variability

➡️ MATLAB output snapshots and summaries are stored in this folder.

---

## 3. Classification Results (Discrete Weight Classes)

**Path:** `results/classification/`

This stage evaluates discrete classification of task weights using supervised learning.

### Model
- **Support Vector Machine (SVM)**
- RBF kernel
- Leave-One-Subject-Out (LOSO) cross-validation

### Weight Classes
- 0 kg  
- 2.5 kg  
- 4.5 kg  
- 7.5 kg  

### Key Results
- Mean Accuracy: **≈ 42.18%**
- Mean Balanced Accuracy: **≈ 41.67%**
- Mid-range weights (2.5 kg, 4.5 kg) classified more reliably
- Higher confusion observed between extreme classes (0 kg, 7.5 kg)

### Interpretation
- EEG patterns encode discriminative information for moderate loads
- Classification difficulty increases for extreme loads due to overlapping neural patterns
- Results highlight the challenge of subject-independent EEG classification

➡️ Confusion matrices and accuracy summaries are available in this folder.

---

## Overall Conclusion

- EEG preprocessing ensured high-quality, reliable signals
- Regression results confirm EEG features capture **continuous task load**
- Classification results demonstrate **moderate but meaningful separability** between load levels
- Combined results validate the feasibility of EEG-based task weight inference under subject-independent settings

This results directory provides a complete experimental validation of the proposed signal processing and machine learning pipeline.


