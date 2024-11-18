import email
import numpy as np
import pandas as pd

df = pd.read_csv("../datasets/emails.csv")
def get_field(field, messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))
    return column


def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column


def employee(file):
    column = []
    for string in file:
        column.append(string.split("/")[0])
    return column


def preprocess_folder(folders):
    column = []
    for folder in folders:
        if (folder is None or folder == ""):
            column.append(np.nan)
        else:
            column.append(folder.split("\\")[-1].lower())
    return column


def replace_empty_with_nan(subject):
    column = []
    for val in subject:
        if (val == ""):
            column.append(np.nan)
        else:
            column.append(val)
    return column


df['date'] = get_field("Date", df['message'])
df['subject'] = get_field("Subject", df['message'])
df['X-Folder'] = get_field("X-Folder", df['message'])
df['X-From'] = get_field("X-From", df['message'])
df['X-To'] = get_field("X-To", df['message'])
df['body'] = body(df['message'])
df['employee'] = employee(df['file'])
df['X-Folder'] = preprocess_folder(df['X-Folder'])
df['subject'] = replace_empty_with_nan(df['subject'])
df['X-To'] = replace_empty_with_nan(df['X-To'])

df.dropna(axis=0, inplace=True)
cols_to_drop = ['file', 'message', 'date', 'X-From', 'X-To', 'employee']
df.drop(cols_to_drop, axis=1, inplace=True)
df1 = df[df['X-Folder'].str.contains('sent', na=False)]
df1.drop(['X-Folder'], axis=1, inplace=True)
df1['label'] = 'ham'
df1.to_csv("./parsed_datasets/enron_cleaned_data.csv", index=False)
