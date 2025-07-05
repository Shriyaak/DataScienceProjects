import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import shap
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Step 1: Data Preprocessing
class DataPreprocessor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def preprocess(self):
        # Drop ID column and extract features
        self.X = self.df.drop(columns=["Class", "Sample code number"])
        
        # Map class labels: 2 -> 0 (benign), 4 -> 1 (malignant)
        self.y = self.df["Class"].replace({2: 0, 4: 1})

        # Scale features
        self.X = self.scaler.fit_transform(self.X)

        # Save the fitted scaler
        joblib.dump(self.scaler, "models/scaler.pkl")
        return self.X, self.y

    def perform_eda(self):
        # Plot and save class distribution
        sns.countplot(x=self.df["Class"].replace({2: "Benign", 4: "Malignant"}))
        plt.title("Class Distribution")
        plt.xlabel("Tumor Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("eda_class_distribution.png")
        plt.close()

# Step 2: Model Training and Evaluation
class ModelTrainer:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.trained_models = {}
        self.accuracies = {}
        self.conf_matrices = {}
        self.cv_scores = {}

    def train_and_evaluate(self):
        for name, model in self.models.items():
            print(f"\n[INFO] Training: {name}")

            # Train
            model.fit(self.X_train, self.y_train)

            # Predict
            y_pred = model.predict(self.X_test)

            # Metrics
            acc = accuracy_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)

            # Cross-validation
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_score = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='accuracy').mean()

            # Print metrics
            print("Test Accuracy:", acc)
            print("Cross-Val Accuracy (5-fold):", cv_score)
            print("Confusion Matrix:\n", cm)
            print("Classification Report:\n", report)

            # Save model
            model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)

            # Store results
            self.trained_models[name] = model
            self.accuracies[name] = acc
            self.cv_scores[name] = cv_score
            self.conf_matrices[name] = cm

        self.plot_accuracy_comparison()
        self.plot_confusion_matrices()

    def plot_accuracy_comparison(self):
        df_plot = pd.DataFrame({
            "Model": list(self.accuracies.keys()),
            "Test Accuracy": list(self.accuracies.values()),
            "CV Accuracy": list(self.cv_scores.values())
        })
        df_plot.set_index("Model")[["Test Accuracy", "CV Accuracy"]].plot(kind="bar", ylim=(0.8, 1.0))
        plt.title("Model Accuracy Comparison (Test vs Cross-Validation)")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig("model_accuracy_comparison.png")
        plt.close()

    def plot_confusion_matrices(self):
        for name, cm in self.conf_matrices.items():
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
            plt.close()

    def get_model(self, model_name):
        return self.trained_models.get(model_name, None), self.X_train, self.X_test

# Step 3: Explainability using SHAP
class Explainability:
    @staticmethod
    def generate_shap_summary(model_name, model, X_train, X_test, feature_names):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test[:100], columns=feature_names)

        print(f"\n[INFO] Generating SHAP summary for {model_name}...")

        # Select explainer type
        if model_name in ["Random Forest", "XGBoost"]:
            # Tree-based models: use TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_df)
            # For binary classification, shap_values is a list with two arrays
            # We use shap_values[1] (positive class) for summary plot
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_test_df, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, show=False)

        elif model_name == "Logistic Regression":
            # Linear model: use LinearExplainer
            explainer = shap.LinearExplainer(model, X_train_df, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_test_df)
            shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, show=False)

        else:
            print("[WARN] SHAP not supported for this model:", model_name)
            return

        plt.tight_layout()
        plt.savefig(f"shap_summary_{model_name.replace(' ', '_').lower()}.png")
        plt.close()

# Entry point
if __name__ == "__main__":
    # Step A: Preprocess
    preprocessor = DataPreprocessor("cleaned_breast_cancer.csv")
    X, y = preprocessor.preprocess()
    preprocessor.perform_eda()

    # Step B: Train and evaluate
    trainer = ModelTrainer(X, y)
    trainer.train_and_evaluate()

    # Step C: SHAP Explainability for all models
    feature_names = preprocessor.df.drop(columns=["Class", "Sample code number"]).columns
    for model_name in trainer.trained_models:
        model = trainer.trained_models[model_name]
        X_train = trainer.X_train
        X_test = trainer.X_test
        Explainability.generate_shap_summary(model_name, model, X_train, X_test, feature_names)

    print("\n[INFO] All models trained, evaluated, cross-validated, and visualized, with SHAP explainability.")


                 
    

            
        
    
    
    

    
   