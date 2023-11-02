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
plt.title("Distribution of Penguin Species")
plt.ylabel("Percentage")
plt.xlabel("Species")

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

base_dt_accuracy = accuracy_score(target_test, base_dt_predictions)
base_dt_report = classification_report(target_test, base_dt_predictions)
base_dt_confusion = confusion_matrix(target_test, base_dt_predictions)

plt.figure(figsize=(20, 10))
plot_tree(base_dt, filled=True, feature_names=features.columns, class_names=target.unique(), rounded=True)
plt.title("Base-DT Decision Tree")
plt.show()

#  ---- Top Decision Tree ---- #

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 15],
    'min_samples_split': [7, 8, 9],
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(features_train, target_train)

best_params = grid_search.best_params_

top_dt = DecisionTreeClassifier(**best_params)
top_dt.fit(features_train, target_train)

top_dt_predictions = top_dt.predict(features_test)

top_dt_accuracy = accuracy_score(target_test, top_dt_predictions)
top_dt_report = classification_report(target_test, top_dt_predictions)
top_dt_confusion = confusion_matrix(target_test, top_dt_predictions)
plt.figure(figsize=(20, 10))
plot_tree(top_dt, filled=True, feature_names=features.columns, class_names=target.unique(), rounded=True)
plt.title("Top-DT Decision Tree")
plt.show()

#  ---- Base MLP ---- #

base_mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', alpha=0.0001, batch_size='auto')

base_mlp.fit(features_train, target_train)

base_mlp_predictions = base_mlp.predict(features_test)

base_mlp_accuracy = accuracy_score(target_test, base_mlp_predictions)
base_mlp_report = classification_report(target_test, base_mlp_predictions)
base_mlp_confusion = confusion_matrix(target_test, base_mlp_predictions)

print("Base-MLP Accuracy:", base_mlp_accuracy)
print("\nClassification Report:\n", base_mlp_report)
print("\nConfusion Matrix:\n", base_mlp_confusion)


# ---- Top MLP ---- #

mlp_param_grid = {
    'activation': ['logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    'solver': ['adam', 'sgd']
}

mlp_grid_search = GridSearchCV(MLPClassifier(), mlp_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

mlp_grid_search.fit(features_train, target_train)

mlp_best_params = mlp_grid_search.best_params_

top_mlp = MLPClassifier(**mlp_best_params)
top_mlp.fit(features_train, target_train)

top_mlp_predictions = top_mlp.predict(features_test)

top_mlp_accuracy = accuracy_score(target_test, top_mlp_predictions)
top_mlp_report = classification_report(target_test, top_mlp_predictions)
top_mlp_confusion = confusion_matrix(target_test, top_mlp_predictions)

print("Top-MLP Accuracy:", top_mlp_accuracy)
print("\nClassification Report:\n", top_mlp_report)
print("\nConfusion Matrix:\n", top_mlp_confusion)
print("\nBest Parameters:\n", mlp_best_params)