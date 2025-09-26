from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def create_logistic_regression_model(seed=42):
    """
    Returns an untrained Logistic Regression model pipeline.
    """
    return Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=seed, max_iter=1000, n_jobs=-1))
    ])

def create_decision_tree_model(seed=42):
    """
    Returns an untrained DecisionTreeClassifier.
    """
    return DecisionTreeClassifier(random_state=seed)

def create_random_forest_model(seed=42):
    """
    Returns an untrained RandomForestClassifier.
    """
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

def create_gradient_boosting_model(seed=42):
    """
    Returns an untrained GradientBoostingClassifier. This is another type of ensemble model.
    """
    return GradientBoostingClassifier(n_estimators=100, random_state=seed)

def create_xgboost_model(seed=42):
    """
    Returns an untrained XGBClassifier.
    """
    return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=seed, n_jobs=-1)

def create_and_return_all_models(seed=42):
    """
    Returns dictionary of every model
    """
    return {
        "logistic_regression": create_logistic_regression_model(seed),
        "decision_tree": create_decision_tree_model(seed),
        "random_forest": create_random_forest_model(seed),
        "gradient_boosting": create_gradient_boosting_model(seed),
        "xgboost": create_xgboost_model(seed),
    }