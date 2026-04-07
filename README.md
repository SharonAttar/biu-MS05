# Iris Classification with Support Vector Machines (SVM)

This project provides a clean and clear implementation for classifying Iris flower species using an SVM model. It covers the entire machine learning pipeline, from data exploration and preprocessing to training and evaluation.

## 1. Project Overview
The Iris dataset contains four features (sepal length, sepal width, petal length, and petal width) for three different species of Iris (Setosa, Versicolor, and Virginica). The goal of this project is to build an SVM model that accurately predicts the species based on these features.

## 2. Key Features
- **Data Preprocessing**: Splits the data into training (80%) and testing (20%) sets.
- **Model Selection**: Uses Support Vector Machines (SVM) for multi-class classification.
- **Evaluation Metrics**: Reports accuracy, confusion matrix, and a detailed classification report.
- **Visualizations**: Provides plots for the confusion matrix and feature relationships.
- **Output Management**: Automatically saves metrics and plots to the `outputs/` folder.

## 3. Project Structure
- `PRD.md`: Project requirements.
- `PLAN.md`: Implementation steps.
- `TODO.md`: Task tracking.
- `main.py`: Main execution script.
- `outputs/`: Folder containing generated results and plots.

## 4. How to Run
1. Ensure you have the required libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
2. Run the `main.py` script:
   ```bash
   python main.py
   ```
3. Check the `outputs/` folder for the results.

## 5. Requirements
To run this project, install the following Python packages:
```bash
pip install pandas scikit-learn matplotlib seaborn
```
