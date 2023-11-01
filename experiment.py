import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)


def convert_to_one_hot(df, categorical_columns):
    
    dummies_df = pd.get_dummies(df, columns=categorical_columns)
    
    for col in dummies_df.columns:
        if dummies_df[col].dtype == 'bool':
            dummies_df[col] = dummies_df[col].astype(int)
    return dummies_df


categorical_columns = ['island', 'sex']  

penguins_df = load_dataset('penguins.csv')

penguins_df_encoded = convert_to_one_hot(penguins_df, categorical_columns)


penguins_df_encoded.to_csv('transformedPenguins.csv', index=False)
