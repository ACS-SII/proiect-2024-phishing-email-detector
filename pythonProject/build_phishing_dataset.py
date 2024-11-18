import pandas as pd

csv_files = [
    './parsed_datasets/enron_cleaned_data.csv',
    './parsed_datasets/nazario_cleaned_data.csv',
    './parsed_datasets/nigerian_frauds_cleaned_data.csv',
    './parsed_datasets/spamassassin_dataset.csv'
]

dataframes = [pd.read_csv(file) for file in csv_files]
concatenated_df = pd.concat(dataframes, ignore_index=True)
concatenated_df.to_csv('./parsed_datasets/combined_phishing_dataset.csv', index=False)

print("Datasets concatenated successfully!")
