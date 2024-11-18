import re
import pandas as pd

def clean_body(body):
    metadata_patterns = [
        r"-----Original Message-----",
        r"From:.*",
        r"Sent:.*",
        r"To:.*",
        r"Subject:.*"
    ]

    combined_pattern = "|".join(metadata_patterns)
    cleaned_body = re.sub(combined_pattern, "", body, flags=re.MULTILINE).strip()

    return cleaned_body

def replace_numeric_label(subject):
    column = []
    for val in subject:
        if val == '1':
            column.append('phishing')
        else:
            column.append('ham')
    return column

df = pd.read_csv("../datasets/Nazario_5.csv")

print(df.head().iloc[0])
cols_to_drop = ['sender', 'receiver', 'date', 'urls']
df.drop(cols_to_drop, axis=1, inplace=True)
df['label'] = 'phishing'
df['body'] = df['body'].apply(clean_body)
df.to_csv("./parsed_datasets/nazario_cleaned_data.csv", index=False)
