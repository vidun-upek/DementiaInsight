import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Setup Paths 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def main():
    print("Starting model evaluation and plot generation...")

    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    model_dir = os.path.join(ROOT_DIR, 'models')
    figures_dir = os.path.join(ROOT_DIR, 'report', 'figures')
    
    # Create directories
    os.makedirs(figures_dir, exist_ok=True)
    
    # LOAD ALL SAVED ARTIFACTS 
    try:
        with open(f'{processed_dir}/model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        with open(f'{processed_dir}/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        X_test_processed = np.load(f'{processed_dir}/X_test_processed.npy')
        y_test = pd.read_csv(f'{processed_dir}/y_test.csv').squeeze()
        
        models = {}
        for name in ['LogisticRegression', 'RandomForest', 'LightGBM']:
            with open(f'{model_dir}/{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model or data files not found.")
        print("Please run preprocessing.py and train.py first.")
        return

    sns.set(style='whitegrid')
    
    # Clean feature names
    clean_feature_names = [name.split('__')[-1] for name in feature_names]

    #PLOT MODEL PERFORMANCE COMPARISON 
    results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    results_long = results_df.melt(id_vars='Model', value_vars=['roc_auc', 'pr_auc'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=results_long, palette='viridis')
    plt.title('Model Performance Comparison (AUC)', fontsize=16)
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(f'{figures_dir}/model_comparison.png')
    print(f"Saved model_comparison.png to {figures_dir}")

    # PLOT ROC CURVES
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'r--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right')
    plt.savefig(f'{figures_dir}/roc_curve_comparison.png')
    print(f"Saved roc_curve_comparison.png to {figures_dir}")

    #PLOT PRECISION-RECALL CURVES
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{name} (PR AUC = {pr_auc:.4f})')
    
    no_skill = y_test.mean()
    plt.plot([0, 1], [no_skill, no_skill], 'r--', label=f'No Skill (AUC = {no_skill:.4f})')
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve', fontsize=16)
    plt.legend(loc='lower left')
    plt.savefig(f'{figures_dir}/pr_curve_comparison.png')
    print(f"Saved pr_curve_comparison.png to {figures_dir}")

    # PLOT FEATURE IMPORTANCE 
    model_to_explain = models['LightGBM']
    importances = model_to_explain.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='rocket')
    plt.title('Top 15 Most Important Features (LightGBM)', fontsize=16)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.savefig(f'{figures_dir}/feature_importance.png')
    print(f"Saved feature_importance.png to {figures_dir}")
    print("Evaluation complete. All figures saved.")

if __name__ == "__main__":
    main()
