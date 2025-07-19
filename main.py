from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from extraction import load_data, preprocess_data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = load_data("./playground-series-s5e7/train.csv")
    x,y = preprocess_data(df)

    regressor = RandomForestClassifier(
        n_estimators=1000,             # Nombre d’arbres dans la forêt
        criterion="log_loss",             # Fonction pour mesurer la qualité d’un split: "gini", "entropy", ou "log_loss"
        max_depth=None,               # Profondeur maximale des arbres
        min_samples_split=2,          # Nb min d’échantillons pour splitter un nœud
        min_samples_leaf=1,           # Nb min d’échantillons dans une feuille
        min_weight_fraction_leaf=0.0, # Fraction min du poids total pour une feuille
        max_features="sqrt",          # Nb max de features pour un split ("sqrt", "log2", float, int, None)
        max_leaf_nodes=None,          # Nb max de feuilles
        min_impurity_decrease=0.0,    # Seuil de réduction d’impureté requis
        bootstrap=True,               # Utiliser bootstrap samples ?
        oob_score=False,              # Utiliser les échantillons hors sac ?
        n_jobs=-1,                  # Nb de cœurs CPU (None=1, -1=autant que possible)
        random_state=None,            # Seed de randomisation
        verbose=0,                    # Niveau de verbosité
        warm_start=False,             # Reprendre l'entraînement d'une forêt existante
        class_weight=None,            # Poids des classes : dict, "balanced", ou None
        ccp_alpha=0.0,                # Paramètre de complexité pour la post-élagage (pruning)
        max_samples=None              # Taille d’échantillons bootstrap (si `bootstrap=True`)
    )

    # Split the data into features and target variable
    y = LabelEncoder().fit_transform(y)
    print("Unique classes in target variable:")
    print(np.unique(y))

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in stratified_kfold.split(x, y):
        x_train_fold = x[train_index.tolist()]
        x_test_fold = x[test_index.tolist()]

        y_train_fold = y[train_index.tolist()]
        y_test_fold = y[test_index.tolist()]

        regressor.fit(x_train_fold, y_train_fold)
        predictions = regressor.predict(x_test_fold)
        # Calculate and print evaluation metrics
        accuracy = accuracy_score(y_test_fold, predictions)
        precision = precision_score(y_test_fold, predictions, average='weighted')
        recall = recall_score(y_test_fold, predictions, average='weighted')
        f1 = f1_score(y_test_fold, predictions, average='weighted')
        roc_auc = roc_auc_score(y_test_fold, predictions, multi_class='ovr')
        logloss = log_loss(y_test_fold, predictions)
        # print
        print(f"Fold Evaluation - Accuracy: {accuracy:.4f}")
        print(f"Fold Evaluation - Precision: {precision:.4f}")
        print(f"Fold Evaluation - Recall: {recall:.4f}")
        print(f"Fold Evaluation - F1 Score: {f1:.4f}")
        print(f"Fold Evaluation - ROC AUC: {roc_auc:.4f}")
        print(f"Fold Evaluation - Log Loss: {logloss:.4f}")
        
        # Grille d'hyperparamètres
        param_grid = {
            'n_estimators': [100, 200,500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        }

        # Grid search avec validation croisée stratifiée
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring='average_precision',  # ou 'f1', 'roc_auc', etc.
            n_jobs=-1,
            verbose=2
        )
        
        # Lancer la recherche
        grid_search.fit(x, y)

        # Meilleurs paramètres
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)
        # Confusion Matrix
        # cm = confusion_matrix(y_test_fold, predictions)
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.title("Confusion Matrix")
        # plt.show()