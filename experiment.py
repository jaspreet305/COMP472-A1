import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


def convert_to_one_hot(df, categorical_columns):
    dummies_df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return dummies_df


def convert_to_categorical_codes(df, categorical_columns):
    for column in categorical_columns:
        cat_column = pd.Categorical(df[column])
        df[column] = cat_column.codes
    return df


penguins_df = pd.read_csv("penguins.csv")

categorical_columns = ["island", "sex"]

penguins_df_1h = convert_to_one_hot(penguins_df, categorical_columns)
penguins_df_cat = convert_to_categorical_codes(penguins_df, categorical_columns)

penguins_df_1h.to_csv("transformed_penguins_1h.csv", index=False)
penguins_df_cat.to_csv("transformed_penguins_categorical.csv", index=False)

species_counts = penguins_df_1h["species"].value_counts(normalize=True) * 100
species_counts.plot(kind="bar", color=["orange", "blue", "limegreen"])
# plt.title("Distribution of Penguin Species")
# plt.ylabel("Percentage")
# plt.xlabel("Species")

features = penguins_df_1h.drop("species", axis=1)
target = penguins_df_1h["species"]
# default split 75-25
features_train, features_test, target_train, target_test = train_test_split(
    features, target
)

#  ---- Base Decision Tree ---- #

base_dt = DecisionTreeClassifier()

base_dt.fit(features_train, target_train)

base_dt_predictions = base_dt.predict(features_test)

# plt.figure(figsize=(20, 10))
# plot_tree(base_dt, filled=True, feature_names=features.columns, class_names=target.unique(), rounded=True)
# plt.title("Base-DT Decision Tree")
# plt.show()

#  ---- Top Decision Tree ---- #

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 15],
    "min_samples_split": [7, 8, 9],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

grid_search.fit(features_train, target_train)

best_params = grid_search.best_params_

top_dt = DecisionTreeClassifier(**best_params)
top_dt.fit(features_train, target_train)

top_dt_predictions = top_dt.predict(features_test)

# plt.figure(figsize=(20, 10))
# plot_tree(top_dt, filled=True, feature_names=features.columns, class_names=target.unique(), rounded=True)
# plt.title("Top-DT Decision Tree")
# plt.show()

#  ---- Base MLP ---- #

base_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation="logistic",
    solver="sgd",
    alpha=0.0001,
    batch_size="auto",
)

base_mlp.fit(features_train, target_train)

base_mlp_predictions = base_mlp.predict(features_test)

# ---- Top MLP ---- #

mlp_param_grid = {
    "activation": ["logistic", "tanh", "relu"],
    "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
    "solver": ["adam", "sgd"],
}

mlp_grid_search = GridSearchCV(
    MLPClassifier(), mlp_param_grid, cv=5, scoring="accuracy", n_jobs=-1
)

mlp_grid_search.fit(features_train, target_train)

mlp_best_params = mlp_grid_search.best_params_

top_mlp = MLPClassifier(**mlp_best_params)
top_mlp.fit(features_train, target_train)

top_mlp_predictions = top_mlp.predict(features_test)


def write_performance_to_file(
    model_name,
    predictions,
    true_values,
    best_params=None,
    filename="penguin-performance.txt",
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

        report = classification_report(true_values, predictions, output_dict=True)
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


write_performance_to_file("Base-DT", base_dt_predictions, target_test)
write_performance_to_file(
    "Top-DT", top_dt_predictions, target_test, best_params=best_params
)
write_performance_to_file("Base-MLP", base_mlp_predictions, target_test)
write_performance_to_file(
    "Top-MLP", top_mlp_predictions, target_test, best_params=mlp_best_params
)

# ---- ABALONE ----


# Function to convert categorical columns to one-hot encoded columns
def convert_to_one_hot(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns, dtype=int)


# Abalone Data Preprocessing
abalone_df = pd.read_csv("abalone.csv")
abalone_df_1h = convert_to_one_hot(abalone_df, ["Type"])
abalone_df_1h.to_csv("transformed_abalone_1h.csv", index=False)

features_abalone = abalone_df_1h.drop(["Type_M", "Type_F", "Type_I"], axis=1)
target_abalone = abalone_df["Type"]
(
    features_train_abalone,
    features_test_abalone,
    target_train_abalone,
    target_test_abalone,
) = train_test_split(features_abalone, target_abalone)

#  ---- Base Decision Tree ---- #

clf_base_dt = DecisionTreeClassifier()
clf_base_dt.fit(features_train_abalone, target_train_abalone)
base_dt_pred = clf_base_dt.predict(features_test_abalone)

# Visualization for Base-DT
plt.figure(figsize=(20, 10))
plot_tree(
    clf_base_dt,
    filled=True,
    feature_names=features_abalone.columns,
    class_names=clf_base_dt.classes_,
    rounded=True,
    max_depth=5,
)
plt.show()

#  ---- Top Decision Tree ---- #

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10],  # Assuming 5 and 10 as the two values of choice
    "min_samples_split": [
        2,
        5,
        10,
    ],  # Assuming 2, 5, and 10 as the three values of choice
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(features_train_abalone, target_train_abalone)

# Print best parameters
print("Best hyperparameters for Top-DT:", grid_search.best_params_)

# Visualize the best decision tree (Top-DT)
best_tree = grid_search.best_estimator_
plt.figure(figsize=(20, 10))
plot_tree(
    best_tree,
    filled=True,
    feature_names=features_abalone.columns,
    class_names=best_tree.classes_,
    rounded=True,
    max_depth=5,  # Restricting depth for visualization
)
plt.show()

# ---- Base MLP ---- #

clf_base_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100), activation="logistic", solver="sgd"
)
clf_base_mlp.fit(features_train_abalone, target_train_abalone)
base_mlp_pred = clf_base_mlp.predict(features_test_abalone)
