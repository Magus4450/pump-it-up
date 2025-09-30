import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_file_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res

def save_file(path, file):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def plot_f1_scores(result_dict):
    # Convert dict → DataFrame
    df = pd.DataFrame(result_dict).T.reset_index()
    df.columns = ["Model", "Train F1", "Test F1"]

    # Melt for seaborn
    df_melted = df.melt(id_vars="Model", value_vars=["Train F1", "Test F1"],
                        var_name="Dataset", value_name="F1 Score")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="F1 Score", hue="Dataset")

    plt.title("Macro F1 Scores by Model")
    plt.ylim(0, 1)  # since F1 is bounded
    plt.xticks(rotation=20)
    plt.legend(title="")

    plt.tight_layout()
    plt.show()