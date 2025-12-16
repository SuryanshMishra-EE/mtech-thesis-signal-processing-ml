# Classification Performance Summary (SVM – LOSO)

This table summarizes classification performance obtained using an SVM classifier
under Leave-One-Subject-Out (LOSO) cross-validation.

## Performance Metrics
- Mean Accuracy: **42.18%**
- Standard Deviation: **15.98%**
- Mean Balanced Accuracy: **41.67%**

## Observations
- Mid-range weights (2.5 kg and 4.5 kg) show higher classification accuracy.
- Extreme weights (0 kg and 7.5 kg) exhibit higher confusion, indicating overlapping EEG patterns.
- Performance variability across folds highlights strong inter-subject differences.

These results indicate moderate discriminability of EEG features for multi-class
weight classification under subject-independent evaluation.
