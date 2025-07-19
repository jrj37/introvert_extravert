
from extraction import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
import numpy as np
import polars as pl

class Training:
    def __init__(self, model: RandomForestClassifier, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the Training class with a model, features, and target labels.
        """
        self.model = model
        self.x = x
        self.y = y

    def train(self) -> RandomForestClassifier:
        """        
        Trains the model using the provided features and target labels.
        """
        self.model.fit(self.x, self.y)
        return self.model

    def inference(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Performs inference on the test set and prints evaluation metrics.
        Args:
            x_test (np.ndarray): Features of the test set.
            y_test (np.ndarray): True labels of the test set.
        """
        predictions = self.model.predict(x_test)
        # Calculate and print evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        roc_auc = roc_auc_score(y_test, predictions, multi_class='ovr')
        logloss = log_loss(y_test, predictions)
        print(f"Evaluation - Accuracy: {accuracy:.4f}")
        print(f"Evaluation - Precision: {precision:.4f}")
        print(f"Evaluation - Recall: {recall:.4f}")
        print(f"Evaluation - F1 Score: {f1:.4f}")
        print(f"Evaluation - ROC AUC: {roc_auc:.4f}")
        print(f"Evaluation - Log Loss: {logloss:.4f}")