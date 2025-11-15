import pandas as pd
import numpy as np
import pickle
import os
import sys
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def main():
    print("Starting model training...")
    
    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    model_dir = os.path.join(ROOT_DIR, 'models')
    
    try:
        X_train_processed = np.load(f'{processed_dir}/X_train_processed.npy')
        X_test_processed = np.load(f'{processed_dir}/X_test_processed.npy')
        y_train = pd.read_csv(f'{processed_dir}/y_train.csv').squeeze()
        y_test = pd.read_csv(f'{processed_dir}/y_test.csv').squeeze()
        with open(f'{processed_dir}/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Processed data not found in {processed_dir}.")
        print("Please run preprocessing.py first.")
        return

    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(
            random_state=42, n_estimators=50, max_depth=10, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbosity=-1)
    }

    results = {}
    os.makedirs(model_dir, exist_ok=True)
    
    clean_feature_names = [name.split('__')[-1] for name in feature_names]
    categorical_cols = [name for name in clean_feature_names if not name in ['NACCAGE', 'EDUC', 'SMOKYRS']]

    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        if name == 'LightGBM':
            model.fit(X_train_processed, y_train, feature_name=clean_feature_names, categorical_feature=categorical_cols)
        else:
            model.fit(X_train_processed, y_train)
        
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"Test ROC AUC: {results[name]['roc_auc']:.4f}")
        print(f"Test PR AUC: {results[name]['pr_auc']:.4f}")
        
        with open(f'{model_dir}/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved trained {name} model to {model_dir}/{name}.pkl\n")

    with open(f'{processed_dir}/model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"All model results saved to {processed_dir}/model_results.pkl")
    print("Model training complete.")

if __name__ == "__main__":
    main()