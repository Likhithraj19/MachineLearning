import pandas as pd

def load_data(file):
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        print("Unsupported file format. Please provide a CSV or Excel file.")
        return
    
    print("Dataset information:")
    print(df.info())
    print("\nTop rows of the dataset:")
    print(df.head(1))

file = 'train.csv'
load_data(file)
