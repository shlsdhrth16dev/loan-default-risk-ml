import os
import pandas as pd

print("CWD:", os.getcwd())
print("List '.' :", os.listdir('.'))
if os.path.exists('data'):
    print("List 'data':", os.listdir('data'))
    if os.path.exists('data/raw'):
        print("List 'data/raw':", os.listdir('data/raw'))
        
        target = 'data/raw/loan_data.csv'
        if os.path.exists(target):
            print(f"File {target} EXISTS.")
            try:
                df = pd.read_csv(target)
                print("Read success. Shape:", df.shape)
            except Exception as e:
                print("Read FAILED:", e)
        else:
            print(f"File {target} DOES NOT EXIST.")
            
        target2 = 'data/raw/Loan_default.csv'
        if os.path.exists(target2):
             print(f"File {target2} EXISTS.")
        else:
             print(f"File {target2} DOES NOT EXIST.")

    else:
        print("'data/raw' not found")
else:
    print("'data' directory not found")
