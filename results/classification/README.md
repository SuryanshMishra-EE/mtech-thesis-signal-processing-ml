# Classification Results (SVM – LOSO)

This folder contains results of EEG-based classification experiments performed
using Support Vector Machine (SVM) models with Leave-One-Subject-Out (LOSO)
cross-validation.

## Task
- Objective: Classify lifted weight levels from EEG features
- Input features: Bandpower features (delta–low gamma)
- Dimensionality reduction: PCA
- Validation scheme: LOSO (subject-wise)

## Model Details
- Classifier: SVM (RBF kernel)
- Class balancing: Enabled
- PCA components: 4

## Evaluation Metrics
- Accuracy (per fold)
- Balanced Accuracy
- Confusion Matrix (aggregated across folds)

## Notes
- Subjects with unreliable signals were excluded prior to training.
- Results reported reflect generalization across unseen subjects.
