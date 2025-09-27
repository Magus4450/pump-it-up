import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from scipy.stats import loguniform, randint

def get_param_dists():
    """
    Returns a dictionary of parameter distributions for hyperparameter tuning.
    The 'classifier__' prefix is essential for targeting parameters within a scikit-learn Pipeline.
    """
    return {
        "logistic_regression": {
            "classifier__C": loguniform(1e-3, 1e2),
        },
        "decision_tree": {
            "classifier__max_depth": [None, 10, 20, 40, 80],
            "classifier__min_samples_leaf": randint(1, 10),
            "classifier__max_features": ["sqrt", "log2", None],
        },
        "random_forest": {
            "classifier__n_estimators": randint(200, 1000),
            "classifier__max_depth": [None, 10, 20, 40, 80],
            "classifier__min_samples_leaf": randint(1, 6),
            "classifier__max_features": ["sqrt", "log2"],
        },
        "gradient_boosting": {
            "classifier__learning_rate": loguniform(1e-3, 3e-1),
            "classifier__n_estimators": randint(150, 601),
            "classifier__max_depth": [3, 6, 9],
        },
        "xgboost": {
            "classifier__learning_rate": loguniform(1e-3, 3e-1),
            "classifier__n_estimators": randint(150, 1000),
            "classifier__max_depth": randint(3, 10),
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
    }

def run_hyperparameter_search(models, param_dists, X, y, n_iter=10):
    """
    Performs RandomizedSearchCV for given models and parameter distributions.
    This function inherently uses cross-validation.
    """
    # This CV object is used by RandomizedSearchCV to split data for each trial.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    scoring = {"f1_macro": "f1_macro", "balanced_acc": "balanced_accuracy"}
    
    best_estimators = {}
    search_results = {}

    print("\n--- Starting Cross-Validated Hyperparameter Search ---")
    for name, model in models.items():
        if name not in param_dists:
            print(f"\nSkipping {name}: No parameter distribution defined.")
            continue
            
        print(f"\nSearching for best parameters for: {name}...")
        dist = param_dists[name]
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=dist,
            n_iter=n_iter,
            scoring=scoring,
            refit="f1_macro", # Refits the best model on the whole data using this score
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        search.fit(X, y)
        
        best_estimators[name] = search.best_estimator_
        search_results[name] = {
            "best_params": search.best_params_,
            "best_f1_macro": search.best_score_,
            "best_balanced_acc": search.cv_results_["mean_test_balanced_acc"][search.best_index_],
        }

    summary_df = pd.DataFrame(search_results).T.sort_values("best_f1_macro", ascending=False)
    print("\n--- Hyperparameter Search Complete ---")
    
    return summary_df, best_estimators