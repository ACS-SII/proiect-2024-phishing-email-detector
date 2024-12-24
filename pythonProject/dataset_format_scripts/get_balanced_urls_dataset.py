import pandas as pd

df = pd.read_csv('../parsed_datasets/balanced_phishing_dataset.csv')

df = df.filter(['num_links', 'extracted_links', 'label'],axis=1)

df = df[df['num_links'] != 0]
print(df['label'].value_counts())

phishing_emails = df[df['label'] == 'phishing']
non_phishing_emails = df[df['label'] == 'ham']

# Create New Dataset (450 Phishing/450 Non-Phishing)
phishing_sample = phishing_emails.sample(n=400, random_state=42)
phishing_sample = phishing_sample.drop_duplicates()
non_phishing_sample = non_phishing_emails.sample(n=400, random_state=42)
df_sample = pd.concat([phishing_sample, non_phishing_sample]).reset_index(drop=True)
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# CSV
df_sample.to_csv('../parsed_datasets/balanced_urls_dataset.csv', index=False)

rows, cols = df_sample.shape
print(f"The dataset contains {rows} rows and {cols} columns.")