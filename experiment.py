import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def convert_to_one_hot(df, categorical_columns):
    dummies_df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return dummies_df

def convert_to_categorical_codes(df, categorical_columns):
    for column in categorical_columns:
        cat_column = pd.Categorical(df[column])
        df[column] = cat_column.codes
    return df

penguins_df = pd.read_csv('penguins.csv')

categorical_columns = ['island', 'sex']  

penguins_df_1h = convert_to_one_hot(penguins_df, categorical_columns)
penguins_df_cat = convert_to_categorical_codes(penguins_df, categorical_columns)

penguins_df_1h.to_csv('transformed_penguins_1h.csv', index=False)
penguins_df_cat.to_csv('transformed_penguins_categorical.csv', index=False)

species_counts = penguins_df_1h['species'].value_counts(normalize=True) * 100
species_counts.plot(kind='bar', color=['orange', 'blue', 'limegreen'])
plt.title('Distribution of Penguin Species')
plt.ylabel('Percentage')
plt.xlabel('Species')

X = penguins_df_1h.drop('species', axis=1)  # Features (drop the target column)
y = penguins_df_1h['species']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) # Split the data into training and test sets
# Print the shapes of the splits to verify the sizes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Optionally, print a few rows of each to verify visually
print("\nSample of X_train:")
print(X_train.head())
print("\nSample of y_train:")
print(y_train.head())
print("\nSample of X_test:")
print(X_test.head())
print("\nSample of y_test:")
print(y_test.head())
