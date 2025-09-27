import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def create_and_return_all_models(X_train, seed=42):
    """
    Initializes and returns a dictionary of all the models to be tested.
    Each model is a self-contained scikit-learn Pipeline.
    """
    
    # Define a preprocessing step to scale numerical features
    # This will be the first step in each model's pipeline
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), numeric_features)],
        remainder='passthrough'  # Keep other (e.g., one-hot encoded) columns
    )

    models = {
        "logistic_regression": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced'))
        ]),
        "decision_tree": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=seed, class_weight='balanced'))
        ]),
        "random_forest": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight='balanced_subsample'))
        ]),
        "gradient_boosting": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=seed))
        ]),
        "xgboost": Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=seed, n_jobs=-1))
        ]),
    }
    return models