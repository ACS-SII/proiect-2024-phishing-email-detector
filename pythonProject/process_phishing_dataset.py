import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import *
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
combined_df = pd.read_csv('./parsed_datasets/combined_phishing_dataset.csv')


def html_to_text(html):
    if isinstance(html, str):  # Check if the html is a string
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ')
        return text
    return ""


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def remove_custom_stopwords(p):
    return remove_stopwords(p, stopwords=stop_words)

# Custom preprocessing filters
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_custom_stopwords, remove_stopwords, strip_short, stem_text]


combined_df['body'] = combined_df['body'].apply(html_to_text)
combined_df.drop_duplicates(subset='body', keep='first', inplace=True)
combined_df = combined_df[combined_df['body'].str.strip() != '']
combined_df = combined_df[combined_df['body'].notna()]
combined_df = combined_df[combined_df['body'].notnull()]
combined_df = combined_df[combined_df['body'].apply(is_english)]
stop_words = stopwords.words('english')
stop_words.extend(["nbsp", "font", "sans", "serif", "bold", "arial", "verdana", "helvetica", "http", "https", "www", "html", "enron", "margin", "spamassassin"])
combined_df['body'] = combined_df['body'].apply(remove_stopwords)
# combined_df['body'] = combined_df['body'].apply(preprocess_string, filters=CUSTOM_FILTERS)
combined_df.to_csv('./parsed_datasets/preprocessed_combined_phishing_dataset.csv', index=False)