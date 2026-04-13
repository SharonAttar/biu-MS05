# 4-Class Iris Classification Project

This project extends the classic Iris classification task by introducing a **synthetic fourth class** named "Unknown". The goal is to evaluate how well a machine learning model can handle added uncertainty and distinguish between established species and a challenging new synthetic category.

## 1. Project Overview
The dataset contains four features (sepal length, sepal width, petal length, and petal width) for:
- 3 Original Species: Setosa, Versicolor, and Virginica.
- **1 Synthetic Class**: "Unknown" (added to test classification boundaries and model robustness).

## 2. Why a Synthetic Class?
Introducing a synthetic fourth class makes the classification task harder. The "Unknown" class is generated with random feature values that reside near the boundaries of the original data, forcing the model to learn more complex decision boundaries and testing its ability to handle out-of-distribution or ambiguous samples.

## 3. Key Features
- **Data Augmentation**: Automatically generates synthetic data for the "Unknown" class.
- **Extended Dataset**: Exports the 4-class data to `iris_extended.csv`.
- **Classification Model**: Uses a Support Vector Machine (SVM) for training.
- **Full Evaluation**: Metrics include accuracy, confusion matrix, and classification report.
- **Visualizations**: Heatmaps and class distribution plots saved to `outputs/`.

## 4. Project Structure
- `PRD.md`: Detailed requirements.
- `PLAN.md`: Implementation steps.
- `TODO.md`: Task tracking.
- `main.py`: Main execution script.
- `iris.csv`: Original dataset.
- `iris_extended.csv`: The augmented dataset.
- `outputs/`: Folder containing generated results and plots.

## 5. How to Run
1. Install requirements:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Run the script:
   ```bash
   python main.py
   ```
3. Results will be saved to the `outputs/` directory.
