import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def main():
    # 1. Load the Iris dataset
    # We'll use sklearn's built-in iris dataset for convenience and reliability.
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 2. Explain the dataset structure: features, labels, and class categories.
    print("Iris Dataset Structure:")
    print(f"- Features: {feature_names}")
    print(f"- Target Labels (Numerical): {set(y)}")
    print(f"- Class Categories: {target_names}")
    print(f"- Data shape: {X.shape}")
    print("-" * 30)

    # 3. Split the data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train an SVM classification model
    # A Support Vector Machine (SVM) with a linear kernel is chosen for this multi-class task.
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # 5. Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # 6. Output: accuracy, confusion matrix, classification report
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Evaluation Result:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    # 7. Create visualizations
    # Define outputs directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visualization A: Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix: SVM Iris Classifier')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Visualization B: Pairplot of Features
    # This helps understand feature relationships and species separation.
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    sns.set_theme(style="ticks")
    pairplot = sns.pairplot(df, hue="species", palette="husl", markers=["o", "s", "D"])
    pairplot.fig.suptitle('Iris Dataset Feature Relationships (Pairplot)', y=1.02)
    pairplot.savefig(os.path.join(output_dir, 'feature_pairplot.png'))
    plt.close()

    # 8. Save outputs (metrics) into an outputs folder
    metrics_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write("Iris Classification - SVM Model Evaluation\n")
        f.write("=" * 45 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(cr)

    print(f"\nSUCCESS: Evaluation results and plots have been saved to: {output_dir}")

if __name__ == "__main__":
    main()
