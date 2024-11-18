import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(filepath, needsScaling):
    df = pd.read_csv(filepath)
    df = df[df['body'].notna()]
    df['body'] = df['body'].astype(str)
    df = df[df['body'].str.strip() != '']
    X_train, X_test, y_train, y_test = train_test_split(
        df['body'], df['label'], test_size=0.2, random_state=42
    )
    stop_words = stopwords.words('english')
    stop_words.extend([
        "nbsp", "font", "sans", "serif", "bold", "arial", "verdana",
        "helvetica", "http", "https", "www", "html", "enron", "margin",
        "spamassassin"
    ])
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    if needsScaling:
        scaler = StandardScaler(with_mean=False)
        X_train_tfidf = scaler.fit_transform(X_train_tfidf)
        X_test_tfidf = scaler.transform(X_test_tfidf)

    return X_train_tfidf, X_test_tfidf, y_train, y_test

