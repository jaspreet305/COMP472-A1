import pandas as pd

def convert_to_one_hot(df, categorical_columns):
    dummies_df = pd.get_dummies(df, columns=categorical_columns, dtype=int)
    return dummies_df

penguins_df = pd.read_csv('penguins.csv')

categorical_columns = ['island', 'sex']  

penguins_df_encoded = convert_to_one_hot(penguins_df, categorical_columns)

penguins_df_encoded.to_csv('transformed_penguins_1h.csv', index=False)
