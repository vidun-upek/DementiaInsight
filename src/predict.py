import pandas as pd
import numpy as np
import pickle
import os
import sys

# Import the helper functions so pickle.load() can find them
from utils import to_numeric, to_string

# Setup Paths 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def get_user_input():
    """Gets the 8 required features from the user via command line."""
    
    print("\n--- Please Enter Your Information ---")
    print("This is a research model and NOT a medical diagnosis.")
    print("Enter 'nan' if you don't know or wish to skip a question.\n")

    questions = {
        'NACCAGE': "What is your current age? (e.g., 68): ",
        'EDUC': "How many total years of education have you completed? (e.g., 16): ",
        'MARISTAT': "What is your marital status? (1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married, 6=Living as married, 9=Other): ",
        'NACCLIVS': "What is your living situation? (1=Lives alone, 2=Lives with spouse/partner, 3=Lives with relative/friend, 4=Group home, 5=Other): ",
        'TOBAC30': "In the last 30 days, have you smoked? (0=No, 1=Yes): ",
        'SMOKYRS': "How many total years have you smoked? (e.g., 12): ",
        'ALCOCCAS': "In the past 3 months, have you consumed alcohol? (0=No, 1=Yes): ",
        'ALCFREQ': "In the past 3 months, how often did you drink? (0=Less than once a month, 1=Once a month, 2=Once a week, 3=A few times a week, 4=Daily): "
    }
    
    user_data = {}
    
    for col, question in questions.items():
        while True:
            val = input(question).strip().lower()
            if val == 'nan' or val == '':
                user_data[col] = np.nan
                break
            else:
                try:
                    user_data[col] = float(val)
                    break
                except ValueError:
                    print("Invalid input. Please enter a number or 'nan'.")
                    
    user_df = pd.DataFrame(user_data, index=[0])
    
    column_order = [
        'NACCAGE', 'EDUC', 'SMOKYRS', 'TOBAC30', 
        'ALCOCCAS', 'ALCFREQ', 'MARISTAT', 'NACCLIVS'
    ]
    # Ensure all columns are present and in the right order
    user_df = user_df.reindex(columns=column_order)
    
    return user_df

def main():
    print("--- Dementia Risk Prediction Tool ---")
    
    try:
        preprocessor_path = os.path.join(ROOT_DIR, 'data', 'processed', 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
            
        model_path = os.path.join(ROOT_DIR, 'models', 'LightGBM.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("Could not find preprocessor.pkl or LightGBM.pkl.")
        print("Please run 'python main.py' from the 'src' directory first to train the models.")
        return
    except AttributeError as e:
        print(f"\n--- ATTRIBUTE ERROR ---")
        print(f"Pickle error: {e}")
        print("This usually means 'preprocessor.pkl' is out of date.")
        print("Please re-run 'python main.py' to rebuild all files.")
        return

    user_df = get_user_input()

    print("\nProcessing your information...")
    user_processed = preprocessor.transform(user_df)

    prediction_proba = model.predict_proba(user_processed)
    dementia_risk = prediction_proba[0, 1] * 100

    print("\n--- PREDICTION RESULT ---")
    print(f"Based on the non-medical data provided, the model estimates the risk of dementia as: {dementia_risk:.2f}%")
    
    print("\n*** DISCLAIMER ***")
    print("This is a research prototype from a hackathon. It is NOT a medical diagnosis.")
    print("Please consult a qualified medical professional for any health concerns.")

if __name__ == "__main__":
    main()