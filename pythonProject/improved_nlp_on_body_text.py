import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import contractions


def categorize_label(email_type):
    if email_type == 'phishing':
        return 1
    else:
        return 0


def preprocess_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = [word for word in text.split() if word.isalpha()]
    words = [contractions.fix(word) for word in words]
    words = [word for word in words if len(word) > 2]
    words = [word.lower() for word in words]
    stop_words = set(stopwords.words('english'))
    unwanted_terms = [
        'enron', 'hpl', 'nom', 'forwarded', 'message', 'subject',
        'nbsp', 'font', 'sans', 'serif', 'bold', 'arial', 'verdana', 'helvetica',
        'html', 'margin', 'spamassassin'
    ]
    stop_words.update(unwanted_terms)
    words = [word for word in words if word not in stop_words]
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

combined_df = pd.read_csv('./parsed_datasets/combined_phishing_dataset.csv')
combined_df['categorized_label'] = combined_df['label'].apply(categorize_label)

# Check distribution of categories
print(combined_df['categorized_label'].value_counts())

# Remove duplicates
combined_df = combined_df.sort_values(by='subject', axis=0, ascending=True)
combined_df = combined_df.drop_duplicates(subset={'subject', 'body'}, keep='first', inplace=False)
print(combined_df['categorized_label'].value_counts())

combined_df['body'] = combined_df['body'].fillna('')
combined_df['cleaned_body'] = combined_df['body'].apply(preprocess_text)

# example = combined_df['body'][0]
# cleaned_example = combined_df['cleaned_body'][0]
# print(example)
# print(cleaned_example)

combined_df = combined_df.sort_values(by='subject', axis=0, ascending=True)
combined_df = combined_df.drop_duplicates(subset={'subject', 'cleaned_body'}, keep='first', inplace=False)
print(combined_df['categorized_label'].value_counts())

def extract_links(text):
    # Find all URLs using a regex pattern
    links = re.findall(r'(https?:\/\/(?:www\.)?[a-zA-Z0-9-]+\.[^\s]{2,}|www\.[a-zA-Z0-9-]+\.\S{2,})', text)
    return links

# Create a new column for extracted links
combined_df['extracted_links'] = combined_df['body'].apply(extract_links)

# Optionally, you can also count the number of links in each email
combined_df['num_links'] = combined_df['extracted_links'].apply(len)

# example = combined_df.iloc[1]
# print("Original Email Body:", example['body'])
# print("Extracted Links:", example['extracted_links'])
# print("Number of Links:", example['num_links'])


combined_df.to_csv('./parsed_datasets/phishing_dataset_vivi.csv', index=False)