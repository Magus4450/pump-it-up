import pandas as pd
from sklearn.metrics import classification_report

def train_model(model, X, y_encoded):
    """Trains a given model on the entire dataset."""
    model_name = model.__class__.__name__
    if 'Pipeline' in model_name:
        model_name = model.steps[-1][1].__class__.__name__

    print(f"\nTraining model: {model_name}...")
    model.fit(X, y_encoded)
    print("Model training complete.")
    return model

def train_and_evaluate_models(models, X_train, y_train, X_val, y_val, label_encoder):
    print("\n--- Training and Evaluating All Models ---")
    
    for name, model in models.items():
        print(f"\n===== {name} =====")
        try:
            model = train_model(model, X_train, y_train)

            predictions_encoded = model.predict(X_val)
            
            y_val_labels = label_encoder.inverse_transform(y_val)
            predictions_labels = label_encoder.inverse_transform(predictions_encoded)
            
            report = classification_report(y_val_labels, predictions_labels)
            print(report)
            
        except Exception as e:
            print(f"Could not train or evaluate {name}. Error: {e}")
    
    print("\n--- Evaluation Complete ---")


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

