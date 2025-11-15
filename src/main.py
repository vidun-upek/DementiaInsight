import os
import time
import sys
import subprocess 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable 

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    start_time = time.time()
    
    # Build an absolute path to the script to run
    script_path = os.path.join(SCRIPT_DIR, script_name)

    command_list = [PYTHON_EXE, script_path]
    
    print(f"Executing: {command_list}")
    
    # Run the command. This will also capture and print errors from the script itself.
    result = subprocess.run(command_list, capture_output=True, text=True, encoding='utf-8')
    
    end_time = time.time()
    
    # Check the exit code
    if result.returncode != 0:
        print(f"*** ERROR: {script_name} failed with exit code {result.returncode} ***")
        print("\n--- ERROR MESSAGE ---")
        print(result.stderr) 
        return False

    print(result.stdout)
    
    print(f"--- Finished {script_name} in {end_time - start_time:.2f} seconds ---")
    return True

def main():
    print("=== STARTING DEMENTIA RISK PREDICTION PIPELINE ===")
    
    if not run_script("preprocessing.py"):
        return
        
    if not run_script("train.py"):
        return
        
    if not run_script("evaluation.py"):
        return
        
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("All models are trained and figures are saved in ../report/figures/")

if __name__ == "__main__":
    main()