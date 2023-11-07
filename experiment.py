import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

def write_performance_to_file(
    model_name,
    predictions,
    true_values,
    filename,
    best_params=None,
):
    with open(filename, "a") as file:
        file.write("- - - - -" * 20 + "\n")
        file.write(f"\n[A]\nModel: {model_name}\n")

        # for top-DT and top-MLP
        if best_params:
            file.write(f"Best Parameters: {best_params}\n")

        file.write("\n[B]\nConfusion Matrix:\n")
        confusion = confusion_matrix(true_values, predictions)
        file.write(str(confusion) + "\n")

        report = classification_report(true_values, predictions, output_dict=True, zero_division=1)
        file.write("\n[C]\nPrecision, Recall, F1-measure for each class:\n")
        for label, metrics in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                file.write(
                    f"Class {label} - Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1-measure: {metrics['f1-score']:.2f}\n"
                )

        file.write("\n[D]\nModel-wide Metrics:\n")
        file.write(f"Accuracy: {report['accuracy']:.2f}\n")
        file.write(f"Macro-average F1: {report['macro avg']['f1-score']:.2f}\n")
        file.write(f"Weighted-average F1: {report['weighted avg']['f1-score']:.2f}\n\n")

def convert_to_one_hot(df, categorical_columns):
    dummies_df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return dummies_df

def convert_to_categorical_codes(df, categorical_columns):
    for column in categorical_columns:
        cat_column = pd.Categorical(df[column])
        df[column] = cat_column.codes
    return df

# - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - P E N G U I N S - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - #

penguins_df = pd.read_csv("datasets/penguins.csv")

categorical_columns = ["island", "sex"]

penguins_df_1h = convert_to_one_hot(penguins_df, categorical_columns)
penguins_df_cat = convert_to_categorical_codes(penguins_df, categorical_columns)

penguins_df_1h.to_csv("transformed_data/penguins_1h.csv", index=False)
penguins_df_cat.to_csv("transformed_data/penguins_categorical.csv", index=False)

species_counts = penguins_df_1h["species"].value_counts(normalize=True) * 100
species_counts.plot(kind="bar", color=["purple", "green", "orange"])
plt.title("Distribution of Penguin Species")
plt.ylabel("Percentage")
plt.xlabel("Species")
plt.savefig('plots/classes-penguin.png', format='png')

features_penguin = penguins_df_1h.drop("species", axis=1)
target_penguin  = penguins_df_1h["species"]

# default split 75-25
features_train_penguin, features_test_penguin, target_train_penguin, target_test_penguin = train_test_split(
    features_penguin, target_penguin
)

#  ---- Penguin: Base Decision Tree ---- #

base_dt_penguin = DecisionTreeClassifier()

base_dt_penguin.fit(features_train_penguin, target_train_penguin)

base_dt_predictions_penguin = base_dt_penguin.predict(features_test_penguin)

plt.figure(figsize=(20, 10))
plot_tree(base_dt_penguin, filled=True, feature_names=features_penguin.columns, class_names=target_penguin.unique(), rounded=True)
plt.title("Base-DT Decision Tree")
plt.savefig('plots/base-dt-penguin.png', format='png')

#  ---- Penguin: Top Decision Tree ---- #

param_grid_penguin_tdt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 15],
    "min_samples_split": [5, 10, 50],
}

grid_search_tdt_penguin = GridSearchCV(DecisionTreeClassifier(), param_grid_penguin_tdt)

grid_search_tdt_penguin.fit(features_train_penguin, target_train_penguin)

best_tree_dt_penguin = grid_search_tdt_penguin.best_estimator_
best_params_dt_penguin = grid_search_tdt_penguin.best_params_
top_dt_predictions_penguin = best_tree_dt_penguin.predict(features_test_penguin)

plt.figure(figsize=(20, 10))
plot_tree(best_tree_dt_penguin, filled=True, feature_names=features_penguin.columns, class_names=target_penguin.unique(), rounded=True)
plt.title("Top-DT Decision Tree")
plt.savefig('plots/top-dt-penguin.png', format='png')

#  ---- Penguin: Base MLP ---- #

base_mlp_penguin = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation="logistic",
    solver="sgd"
)

base_mlp_penguin.fit(features_train_penguin, target_train_penguin)

base_mlp_predictions_penguin = base_mlp_penguin.predict(features_test_penguin)

# ---- Penguin: Top MLP ---- #

# for this, default number of iterations is 200, so optimization may not converge all the time.
param_grid_penguin_mlp = {
    "activation": ["logistic", "tanh", "relu"],
    "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
    "solver": ["adam", "sgd"]
}

grid_search_mlp_penguin = GridSearchCV(MLPClassifier(), param_grid_penguin_mlp)

grid_search_mlp_penguin.fit(features_train_penguin, target_train_penguin)

top_mlp_penguin = grid_search_mlp_penguin.best_estimator_
best_params_mlp_penguin = grid_search_mlp_penguin.best_params_
top_mlp_predictions_penguin = top_mlp_penguin.predict(features_test_penguin)

# write to performance files
write_performance_to_file("Base-DT", base_dt_predictions_penguin, target_test_penguin, "performance/penguin-performance.txt")
write_performance_to_file("Top-DT", top_dt_predictions_penguin, target_test_penguin, "performance/penguin-performance.txt", best_params=best_params_dt_penguin)
write_performance_to_file("Base-MLP", base_mlp_predictions_penguin, target_test_penguin, "performance/penguin-performance.txt")
write_performance_to_file("Top-MLP", top_mlp_predictions_penguin, target_test_penguin, "performance/penguin-performance.txt", best_params=best_params_mlp_penguin)

# - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - A B A L O N E - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - #

abalone_df = pd.read_csv("datasets/abalone.csv")

abalone_df_1h = convert_to_one_hot(abalone_df, ["Type"])
abalone_df_1h.to_csv("transformed_data/abalone_1h.csv", index=False)

features_abalone = abalone_df_1h.drop(["Type_M", "Type_F", "Type_I"], axis=1)
target_abalone = abalone_df["Type"]

features_train_abalone, features_test_abalone, target_train_abalone, target_test_abalone = train_test_split(
    features_abalone, target_abalone
)

class_count = target_abalone.value_counts(normalize=True) * 100

plt.figure(figsize=(20, 10))
class_count.plot(kind='bar', color=['purple', 'green', 'orange'])
plt.title('Distribution of Abalone Classes')
plt.ylabel('Percentage')
plt.xlabel('Class')
plt.savefig('plots/classes-abalone.png', format='png')

# ---- Abalone: Base Decision Tree ---- #

base_dt_abalone = DecisionTreeClassifier()
base_dt_abalone.fit(features_train_abalone, target_train_abalone)
base_dt_predictions_abalone = base_dt_abalone.predict(features_test_abalone)

plt.figure(figsize=(20, 10))
plot_tree(
    base_dt_abalone,
    filled=True,
    feature_names=features_abalone.columns,
    class_names=base_dt_abalone.classes_,
    rounded=True,
    max_depth=5,
)
plt.savefig('plots/base-dt-abalone.png', format='png')

#  ---- Abalone: Top Decision Tree ---- #

param_grid_abalone_tdt = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 15],
    "min_samples_split": [5, 10, 50],
}

grid_search_tdt_abalone = GridSearchCV(DecisionTreeClassifier(), param_grid_abalone_tdt)

grid_search_tdt_abalone.fit(features_train_abalone, target_train_abalone)

best_tree_dt_abalone = grid_search_tdt_abalone.best_estimator_
best_params_dt_abalone = grid_search_tdt_abalone.best_params_
top_dt_predictions_abalone = best_tree_dt_abalone.predict(features_test_abalone)

plt.figure(figsize=(20, 10))
plot_tree(
    best_tree_dt_abalone,
    filled=True,
    feature_names=features_abalone.columns,
    class_names=best_tree_dt_abalone.classes_,
    rounded=True,
    max_depth=5,
)
plt.savefig('plots/top-dt-abalone.png', format='png')

# ---- Base MLP ---- #

base_mlp_abalone = MLPClassifier(
    hidden_layer_sizes=(100, 100), 
    activation="logistic", 
   solver="sgd"
)
base_mlp_abalone.fit(features_train_abalone, target_train_abalone)
base_mlp_predictions_abalone = base_mlp_abalone.predict(features_test_abalone)

# ---- Abalone: Top MLP ---- #

# for this, default number of iterations is 200, so optimization may not converge all the time.
param_grid_abalone_mlp = {
    "activation": ["logistic", "tanh", "relu"],
    "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
    "solver": ["adam", "sgd"],
}

grid_search_mlp_abalone = GridSearchCV(MLPClassifier(), param_grid_abalone_mlp)

grid_search_mlp_abalone.fit(features_train_abalone, target_train_abalone)

top_mlp_abalone = grid_search_mlp_abalone.best_estimator_
best_params_mlp_abalone = grid_search_mlp_abalone.best_params_
top_mlp_predictions_abalone = top_mlp_abalone.predict(features_test_abalone)

# write to performance files
write_performance_to_file("Base-DT", base_dt_predictions_abalone, target_test_abalone, "performance/abalone-performance.txt")
write_performance_to_file("Top-DT", top_dt_predictions_abalone, target_test_abalone, "performance/abalone-performance.txt", best_params=best_params_dt_abalone)
write_performance_to_file("Base-MLP", base_mlp_predictions_abalone, target_test_abalone, "performance/abalone-performance.txt")
write_performance_to_file("Top-MLP", top_mlp_predictions_abalone, target_test_abalone, "performance/abalone-performance.txt", best_params=best_params_mlp_abalone)

# evaluate multiple times
def evaluate_model(model, features_train, target_train, features_test, target_test):
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
    macro_f1 = f1_score(target_test, predictions, average='macro', zero_division=1)
    weighted_f1 = f1_score(target_test, predictions, average='weighted', zero_division=1)
    return accuracy, macro_f1, weighted_f1

# Function to run the evaluation multiple times and calculate statistics
def run_evaluation(model, features_train, target_train, features_test, target_test, iterations=5):
    accuracies = []
    macro_f1s = []
    weighted_f1s = []
    
    for _ in range(iterations):
        accuracy, macro_f1, weighted_f1 = evaluate_model(model, features_train, target_train, features_test, target_test)
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        weighted_f1s.append(weighted_f1)
    
    accuracy_avg = np.mean(accuracies)
    accuracy_var = np.var(accuracies)
    macro_f1_avg = np.mean(macro_f1s)
    macro_f1_var = np.var(macro_f1s)
    weighted_f1_avg = np.mean(weighted_f1s)
    weighted_f1_var = np.var(weighted_f1s)
    
    return accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var

def append_to_performance_file(filename, model_name, accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var):
    with open(filename, "a") as file:
        file.write(f"\n{model_name} - Evaluation over 5 iterations:\n")
        file.write(f"Average Accuracy: {accuracy_avg:.2f}, Variance: {accuracy_var:.2f}\n")
        file.write(f"Average Macro F1: {macro_f1_avg:.2f}, Variance: {macro_f1_var:.2f}\n")
        file.write(f"Average Weighted F1: {weighted_f1_avg:.2f}, Variance: {weighted_f1_var:.2f}\n")

# -- -- -- -- --  PENGUIN EVALUATIONS -- -- -- -- -- 

# Base Decision Tree
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    base_dt_penguin, features_train_penguin, target_train_penguin, features_test_penguin, target_test_penguin
)
append_to_performance_file("performance/penguin-performance.txt", "Base-DT", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# Base MLP
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    base_mlp_penguin, features_train_penguin, target_train_penguin, features_test_penguin, target_test_penguin
)
append_to_performance_file("performance/penguin-performance.txt", "Base-MLP", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# Top Decision Tree
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    best_tree_dt_penguin, features_train_penguin, target_train_penguin, features_test_penguin, target_test_penguin
)
append_to_performance_file("performance/penguin-performance.txt", "Top-DT", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# Top MLP
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    top_mlp_penguin, features_train_penguin, target_train_penguin, features_test_penguin, target_test_penguin
)
append_to_performance_file("performance/penguin-performance.txt", "Top-MLP", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# -- -- -- -- --  ABALONE EVALUATIONS -- -- -- -- -- 

# Base Decision Tree
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    base_dt_abalone, features_train_abalone, target_train_abalone, features_test_abalone, target_test_abalone
)
append_to_performance_file("performance/abalone-performance.txt", "Base-DT", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# Base MLP
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    base_mlp_abalone, features_train_abalone, target_train_abalone, features_test_abalone, target_test_abalone
)
append_to_performance_file("performance/abalone-performance.txt", "Base-MLP", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# Top Decision Tree
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    best_tree_dt_abalone, features_train_abalone, target_train_abalone, features_test_abalone, target_test_abalone
)
append_to_performance_file("performance/abalone-performance.txt", "Top-DT", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)

# Top MLP
accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var = run_evaluation(
    top_mlp_abalone, features_train_abalone, target_train_abalone, features_test_abalone, target_test_abalone
)
append_to_performance_file("performance/abalone-performance.txt", "Top-MLP", accuracy_avg, accuracy_var, macro_f1_avg, macro_f1_var, weighted_f1_avg, weighted_f1_var)
