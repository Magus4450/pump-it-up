import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def show_final_evaluation(winner_model, X_train, y_train, label_encoder):
    """
    Fits the winning model on the full training data and shows diagnostic reports.
    """
    print("\n--- Final Model Evaluation on Full Training Data ---")
    
    # The winner_model from RandomizedSearchCV is already fitted on the full data.
    # We just need to make predictions.
    train_pred_encoded = winner_model.predict(X_train)
    
    # Decode for readable report
    y_train_labels = label_encoder.inverse_transform(y_train)
    train_pred_labels = label_encoder.inverse_transform(train_pred_encoded)
    
    print("\nConfusion Matrix (Full Train Set):")
    print(confusion_matrix(y_train_labels, train_pred_labels))
    
    print("\nClassification Report (Full Train Set):")
    print(classification_report(y_train_labels, train_pred_labels, digits=3))
    
    return winner_model # Return the fitted model for submission

def create_submission_file(model, X_test, original_test_df, label_encoder, filename="submission.csv"):
    """
    Uses a trained model to make predictions on the test set and saves the file.
    """
    print(f"\n--- Generating Submission File: {filename} ---")
    
    test_pred_encoded = model.predict(X_test)
    test_pred_labels = label_encoder.inverse_transform(test_pred_encoded)
    
    submission_df = pd.DataFrame({
        'id': original_test_df['id'],
        'status_group': test_pred_labels
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"Submission file saved successfully!")
    
    return submission_df

