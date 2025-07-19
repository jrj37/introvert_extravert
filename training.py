
from sklearn.calibration import LabelEncoder
from extraction import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
import numpy as np

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
    
    def cross_validate(self, n_splits: int = 5) -> RandomForestClassifier:
        """
        Performs cross-validation on the model and prints evaluation metrics for each fold.
        Args:
            n_splits (int): Number of splits for cross-validation.
        """
        stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_index, test_index in stratified_kfold.split(self.x, self.y):
            x_train_fold = self.x[train_index.tolist()]
            x_test_fold = self.x[test_index.tolist()]

            y_train_fold = self.y[train_index.tolist()]
            y_test_fold = self.y[test_index.tolist()]

            self.model.fit(x_train_fold, y_train_fold)
            self.inference(x_test_fold, y_test_fold)
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
        # Calculate and print evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        average_precision = precision_score(y_test, predictions, average='macro')
        roc_auc = roc_auc_score(y_test, predictions, multi_class='ovr')
        logloss = log_loss(y_test, predictions)
        print(f"Fold Evaluation - Accuracy: {accuracy:.4f}")
        print(f"Fold Evaluation - Precision: {precision:.4f}")
        print(f"Fold Evaluation - Recall: {recall:.4f}")
        print(f"Fold Evaluation - F1 Score: {f1:.4f}")
        print(f"Fold Evaluation - Average Precision: {average_precision:.4f}")
        print(f"Fold Evaluation - ROC AUC: {roc_auc:.4f}")
        print(f"Fold Evaluation - Log Loss: {logloss:.4f}")
    