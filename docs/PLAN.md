# Implementation Plan - 4-Class Iris Classification

## Phase 1: Environment Setup
1. Define the project structure.
2. Ensure `iris.csv` exists (if not, create it from `sklearn` for consistency).

## Phase 2: Synthetic Data Generation
1. Load `iris.csv`.
2. Analyze the range of features for existing classes (mean, standard deviation).
3. Generate 50 synthetic rows for a new class **"Unknown"**.
4. Use random values for the "Unknown" class to make it "challenging" but relevant.
5. Append these rows and save the result as `iris_extended.csv`.

## Phase 3: Model Development
1. Split `iris_extended.csv` into 80% Training and 20% Testing.
2. Train a classification model (e.g., SVM).
3. Predict results for the test set.

## Phase 4: Evaluation & Visualization
1. Metrics: Accuracy, Confusion Matrix, Classification Report.
2. Plots:
   - Confusion Matrix heatmap.
   - Class distribution bar chart.
   - Scatter plot showing original vs. synthetic classes in 2D space.
3. Save all outputs to the `outputs/` folder.

## Phase 5: Finalization
1. Update `README.md` and `TODO.md`.
2. Add comprehensive comments to `main.py`.
