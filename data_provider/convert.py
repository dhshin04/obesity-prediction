import pandas as pd

# Replace with your file paths
xlsx_file = "./dataset/obesity_dataset_patient.xlsx"
csv_file = "./dataset/obesity_dataset_patient.csv"

df = pd.read_excel(xlsx_file)
df.to_csv(csv_file, index=False)