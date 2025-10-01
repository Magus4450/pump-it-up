import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from ..preprocessing.load import get_processed_data
from .evaluate import evaluate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from .utils import load_file_if_exists, save_file
import os


def train_and_evaluate_baseline(preset_name, model_name, seed=42, model_kwargs={}):

    res = load_file_if_exists("results.pkl") or {}

    if model_name in res:
        print("Found saved F1s. Returning them.")
        model = load_file_if_exists(f"models/{model_name}.pkl")
        return *res[model_name], model
    
    os.makedirs("models", exist_ok=True)


    train_processed, test_processed, processor = get_processed_data(preset_name=preset_name)
    le = processor.target_le
    X = train_processed.drop("status_group", axis=1)
    y = train_processed["status_group"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)


    lr = LogisticRegression(**model_kwargs)
    lr.fit(X_train, y_train)

    train_f1, test_f1 = evaluate(lr, X_train, y_train, X_test, y_test, le, model_name)

    save_file(f"models/{model_name}.pkl", lr)
    res[model_name] = (train_f1, test_f1)
    save_file(f"results.pkl", res)

    return train_f1, test_f1, lr

def train_w_hpam_search_and_eval(model, preset_name, model_name, param_dist, seed=42, n_cols=10):

    res = load_file_if_exists("results.pkl") or {}

    if model_name in res:
        print("Found saved F1s. Returning them.")
        model = load_file_if_exists(f"models/{model_name}.pkl")
        return *res[model_name], model

    os.makedirs("models", exist_ok=True)


    
    train_processed, test_processed, processor = get_processed_data(preset_name=preset_name, n_cols=n_cols)
    le = processor.target_le

    X = train_processed.drop("status_group", axis=1)
    y = train_processed["status_group"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,                
        scoring="f1_macro",       
        cv=5,                    
        random_state=seed,
        n_jobs=-1,                
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    train_f1, test_f1 = evaluate(best_model, X_train, y_train, X_test, y_test, le, model_name)

    save_file(f"models/{model_name}.pkl", best_model)
    res[model_name] = (train_f1, test_f1)

    save_file(f"results.pkl", res)

    # y_pred = best_model.predict(test_processed.drop(columns=["id"]))

    # # Build dataframe with id and predicted labels
    # pred_df = pd.DataFrame({
    #     "id": test_processed["id"].values,
    #     "status_group": y_pred
    # })
    # pred_df["status_group"] = processor.target_le.inverse_transform(pred_df["status_group"])
    # pred_df.to_csv(f"{model_name}_sub.csv", index=False)

    # pred_df.head()
   
    return train_f1, test_f1, best_model


# def train_model(model, X, y_encoded):
#     """Trains a given model on the entire dataset."""
#     model_name = model.__class__.__name__
#     if 'Pipeline' in model_name:
#         model_name = model.steps[-1][1].__class__.__name__

#     print(f"\nTraining model: {model_name}...")
#     model.fit(X, y_encoded)
#     print("Model training complete.")
#     return model

# def train_and_evaluate_models(models, X_train, y_train, X_val, y_val, label_encoder):
#     print("\n--- Training and Evaluating All Models ---")
    
#     for name, model in models.items():
#         print(f"\n===== {name} =====")
#         try:
#             model = train_model(model, X_train, y_train)

#             predictions_encoded = model.predict(X_val)
            
#             y_val_labels = label_encoder.inverse_transform(y_val)
#             predictions_labels = label_encoder.inverse_transform(predictions_encoded)
            
#             report = classification_report(y_val_labels, predictions_labels)
#             print(report)
            
#         except Exception as e:
#             print(f"Could not train or evaluate {name}. Error: {e}")
    
#     print("\n--- Evaluation Complete ---")


# def create_submission_file(model, preprocessor, test_df, label_encoder, filename="submission.csv"):
#     """
#     Uses a trained model to make predictions and saves the submission file.
#     """
#     print(f"\nGenerating submission file: {filename}...")
    
#     test_processed = preprocessor.transform(test_df)
#     X_test = test_processed.drop(columns=['id', 'date_recorded'], errors='ignore')
    
#     predictions_encoded = model.predict(X_test)
#     predictions_labels = label_encoder.inverse_transform(predictions_encoded)
    
#     submission_df = pd.DataFrame({
#         'id': test_df['id'],
#         'status_group': predictions_labels
#     })
    
#     submission_df.to_csv(filename, index=False)
#     print(f"Submission file saved successfully to '{filename}'!")
    
#     return submission_df

