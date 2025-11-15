import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import the helper functions from our new utils file
from utils import to_numeric, to_string

#Setup Paths 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def main():
    print("Starting preprocessing...")
    
    na_values = [-4, 9, 88, 99, 888, 999, 9999]
    raw_data_path = os.path.join(ROOT_DIR, 'data', 'raw', 'dataset.csv')
    
    try:
        df = pd.read_csv(raw_data_path, na_values=na_values, low_memory=False)
    except FileNotFoundError:
        print(f"Error: dataset.csv not found at {raw_data_path}")
        return

    non_medical_features = [
        'NACCAGE', 'EDUC', 'MARISTAT', 'NACCLIVS', 
        'TOBAC30', 'SMOKYRS', 'ALCOCCAS', 'ALCFREQ'
    ]
    target = 'DEMENTED'
    
    X = df[non_medical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_features = ['NACCAGE', 'EDUC', 'SMOKYRS']
    categorical_features = ['TOBAC30', 'ALCOCCAS', 'ALCFREQ', 'MARISTAT', 'NACCLIVS']

    numeric_transformer = Pipeline(steps=[
        ('to_numeric', FunctionTransformer(to_numeric, feature_names_out='one-to-one')), 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('to_string', FunctionTransformer(to_string, feature_names_out='one-to-one')),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    
    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(f'{processed_dir}/X_train_processed.npy', X_train_processed)
    np.save(f'{processed_dir}/X_test_processed.npy', X_test_processed)
    y_train.to_csv(f'{processed_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{processed_dir}/y_test.csv', index=False)
    

    with open(f'{processed_dir}/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(f'{processed_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    print(f"Preprocessing complete. Processed data saved to {processed_dir}")

if __name__ == "__main__":
    main()