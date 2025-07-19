import polars as pl
from sklearn.ensemble import RandomForestClassifier
from extraction import load_data, preprocess_data
from sklearn.preprocessing import LabelEncoder
from training import Training

if __name__ == "__main__":
    df = load_data("../playground-series-s5e7/train.csv")
    x,y = preprocess_data(df)
    #Best parameters: {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}

    regressor = RandomForestClassifier(
        n_estimators=200,            # Nombre d’arbres dans la forêt
        criterion="gini",             # Fonction pour mesurer la qualité d’un split: "gini", "entropy", ou "log_loss"
        max_depth=10,               # Profondeur maximale des arbres
        min_samples_split=5,          # Nb min d’échantillons pour splitter un nœud
        min_samples_leaf=2,           # Nb min d’échantillons dans une feuille
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
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    training = Training(model=regressor, x=x, y=y)
    model = training.cross_validate(n_splits=5)
    df = load_data("../playground-series-s5e7/test.csv")
    x,id = preprocess_data(df, train=False)
    preds = model.predict(x)
    preds = label_encoder.inverse_transform(preds)
    print("Predictions:")
    print(preds)
    pl.DataFrame({
        "id": id,
        "Personality": preds
    }).write_csv("../playground-series-s5e7/pred.csv")