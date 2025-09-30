from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate(model, X_train, y_train, X_test, y_test, le, model_name):

    os.makedirs("output", exist_ok=True)

    sns.set_theme('paper')

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_f1 = f1_score(y_train, train_pred, average="macro")
    test_f1  = f1_score(y_test,  test_pred,  average="macro")

    labels = np.arange(len(le.classes_))

    cm_train = confusion_matrix(y_train, train_pred, labels=labels)
    cm_test  = confusion_matrix(y_test,  test_pred,  labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title("Train Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
    axes[1].set_title("Test Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    handles = [plt.Line2D([0], [0], marker="s", color="w", label=f"{i} → {cls}",
                          markerfacecolor="lightgray", markersize=10)
               for i, cls in enumerate(le.classes_)]

    fig.legend(handles=handles, loc="upper center", ncol=len(le.classes_), frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(f"output/{model_name}.png")
    plt.show()

    return train_f1, test_f1

# def show_final_evaluation(winner_model, X_train, y_train, label_encoder):
#     """
#     Fits the winning model on the full training data and shows diagnostic reports.
#     """
#     print("\n--- Final Model Evaluation on Full Training Data ---")
    
#     # The winner_model from RandomizedSearchCV is already fitted on the full data.
#     # We just need to make predictions.
#     train_pred_encoded = winner_model.predict(X_train)
    
#     # Decode for readable report
#     y_train_labels = label_encoder.inverse_transform(y_train)
#     train_pred_labels = label_encoder.inverse_transform(train_pred_encoded)
    
#     print("\nConfusion Matrix (Full Train Set):")
#     print(confusion_matrix(y_train_labels, train_pred_labels))
    
#     print("\nClassification Report (Full Train Set):")
#     print(classification_report(y_train_labels, train_pred_labels, digits=3))
    
#     return winner_model # Return the fitted model for submission

# def create_submission_file(model, X_test, original_test_df, label_encoder, filename="submission.csv"):
#     """
#     Uses a trained model to make predictions on the test set and saves the file.
#     """
#     print(f"\n--- Generating Submission File: {filename} ---")
    
#     test_pred_encoded = model.predict(X_test)
#     test_pred_labels = label_encoder.inverse_transform(test_pred_encoded)
    
#     submission_df = pd.DataFrame({
#         'id': original_test_df['id'],
#         'status_group': test_pred_labels
#     })
    
#     submission_df.to_csv(filename, index=False)
#     print(f"Submission file saved successfully!")
    
#     return submission_df

