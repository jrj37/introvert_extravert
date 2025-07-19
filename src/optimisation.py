from sklearn.model_selection import GridSearchCV,StratifiedKFold

param_grid = {
            'n_estimators': [100, 200,500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
        }

class Optimisation:
    def __init__(self, model, param_grid, scoring='f1') -> None:
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.best_params = None
        self.best_score = None

    def perform_grid_search(self, x, y) -> None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(x, y)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        print("Best parameters:", self.best_params)
        print("Best score:", self.best_score)