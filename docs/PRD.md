# Product Requirements Document (PRD) - Iris Classification (Extended)

## 1. Objective
Build an enhanced Iris classification project that uses a 4-class dataset (3 original species + 1 synthetic class). This project demonstrates model robustness and how classifiers handle challenging, out-of-distribution synthetic data.

## 2. Requirements
- **Dataset**:
  - Source: Local `iris.csv` file.
  - New Class: Add a 4th synthetic class named **"Unknown"**.
  - Generated Data: The "Unknown" class should have realistic features (sepal/petal dimensions) that overlap or lie on the boundaries of the original 3 classes to increase classification difficulty.
  - Save: Export the modified dataset to `iris_extended.csv`.
- **Model**: SVM or another robust classifier for 4-class classification.
- **Split**: 80% Training, 20% Testing.
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- **Visualizations**:
  - Confusion Matrix Plot
  - Class Distribution Plot (Bar chart showing the number of samples per class).
  - Feature comparison (Scatter plot/Box plot).
- **Outputs**: Save results and plots to an `outputs/` directory.

## 3. Project Structure
- `README.md`: Updated project overview including the synthetic class explanation.
- `PRD.md`: This document.
- `PLAN.md`: Implementation strategy.
- `TODO.md`: Task tracking.
- `main.py`: Updated script for data generation, training, and evaluation.
- `iris.csv`: Original dataset.
- `iris_extended.csv`: The 4-class dataset.
- `outputs/`: Folder containing generated plots and metrics.

## 4. Success Criteria
- Successfully generate and integrate the "Unknown" class.
- Achieve a clear classification performance, noting the impact of the synthetic class.
- All required visualizations are generated and saved correctly.
- Well-documented code and project structure.
