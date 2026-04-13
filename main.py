import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    # 1. Load or Create the local iris.csv file
    # If the file doesn't exist, we generate it from sklearn's built-in dataset.
    dataset_path = 'iris.csv'
    if not os.path.exists(dataset_path):
        iris_sk = datasets.load_iris()
        # Create a DataFrame with the original 3 classes
        df_original = pd.DataFrame(data=iris_sk.data, columns=iris_sk.feature_names)
        df_original['species'] = [iris_sk.target_names[i] for i in iris_sk.target]
        df_original.to_csv(dataset_path, index=False)
        print(f"Created initial {dataset_path} with 3 original classes.")
    
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # 2. Inspect the dataset structure
    # Identifying features and target column
    print("Original Dataset structure:")
    print(df.head())
    print(f"Original Classes: {df['species'].unique()}")
    print("-" * 30)

    # 3. Create a synthetic fourth class named "Unknown"
    # This class is added to make the classification task harder and test model uncertainty.
    num_samples = 50
    np.random.seed(42)
    
    # Analyze original feature ranges to generate "realistic" but distinct synthetic data
    means = df.iloc[:, :-1].mean().values
    stds = df.iloc[:, :-1].std().values
    
    synthetic_rows = []
    for _ in range(num_samples):
        # Generate synthetic features by adding random noise to the overall means.
        # This creates a class that overlaps with the original data boundaries.
        row = [
            np.random.uniform(means[0] - stds[0], means[0] + stds[0] * 1.5),
            np.random.uniform(means[1] - stds[1], means[1] + stds[1] * 1.5),
            np.random.uniform(means[2] - stds[2], means[2] + stds[2] * 1.5),
            np.random.uniform(means[3] - stds[3], means[3] + stds[3] * 1.5)
        ]
        synthetic_rows.append(row + ["Unknown"])
    
    # Create DataFrame for the synthetic class
    df_synthetic = pd.DataFrame(synthetic_rows, columns=df.columns)

    # 4. Append synthetic class rows to original dataset
    df_extended = pd.concat([df, df_synthetic], ignore_index=True)
    
    # Save the new 4-class dataset
    extended_path = 'iris_extended.csv'
    df_extended.to_csv(extended_path, index=False)
    print(f"Extended dataset with synthetic 'Unknown' class saved to {extended_path}.")

    # 5. Split the extended dataset into 80% train and 20% test
    X = df_extended.iloc[:, :-1]
    y = df_extended['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train a classification model (SVM)
    # Using a linear kernel to see how it handles the newly overlapping class.
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # 7. Evaluate the model on the test set
    y_pred = model.predict(X_test)
    
    # Generate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    unique_species = sorted(df_extended['species'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=unique_species)
    cr = classification_report(y_test, y_pred)

    print("\nModel Evaluation Results (4 Classes):")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(cr)

    # 8. Save outputs into an outputs folder
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save metrics text file
    with open(os.path.join(output_dir, 'evaluation_metrics_extended.txt'), 'w') as f:
        f.write("Evaluation Results for 4-Class Iris (Including Synthetic 'Unknown')\n")
        f.write("=" * 65 + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cr)

    # 9. Create Visualizations
    # A. Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_species)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix: Iris Extended (4 Classes)')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_extended.png'))
    plt.close()

    # B. Class Distribution Graph
    plt.figure(figsize=(8, 6))
    sns.countplot(x='species', data=df_extended, palette='viridis')
    plt.title('Class Distribution (Original + Synthetic "Unknown")')
    plt.ylabel('Number of Samples')
    plt.savefig(os.path.join(output_dir, 'class_distribution_extended.png'))
    plt.close()

    print(f"\nAll visualizations and metrics have been saved to: {output_dir}/")

if __name__ == "__main__":
    main()
