# Product Requirements Document (PRD) - Iris Classification

## 1. Objective
Build a simple, clear, and ready-to-use Iris classification system using Support Vector Machines (SVM). This project aims to demonstrate data analysis, model training, evaluation, and visualization.

## 2. Requirements
- **Dataset**: Use the Iris dataset (features: sepal length, sepal width, petal length, petal width; labels: species).
- **Model**: Support Vector Machine (SVM) classifier.
- **Split**: 80% Training, 20% Testing.
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- **Visualizations**:
  - Confusion Matrix Plot
  - Feature Distribution/Pairplot or PCA visualization.
- **Outputs**: Save results and plots to an `outputs/` directory.

## 3. Project Structure
- `README.md`: Project overview and usage.
- `PRD.md`: This document.
- `PLAN.md`: Implementation strategy.
- `TODO.md`: Task tracking.
- `main.py`: The core script for training and evaluation.
- `outputs/`: Folder containing generated plots and metrics.

## 4. Success Criteria
- The model achieves high accuracy on the test set.
- All required visualizations are generated and saved correctly.
- The project is well-documented and follows clean code practices.
