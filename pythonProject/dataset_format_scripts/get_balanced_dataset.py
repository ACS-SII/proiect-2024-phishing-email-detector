import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv('../parsed_datasets/phishing_dataset_vivi.csv')
df = df.dropna()

# Dropping N/A Values
print(df.isnull().sum())

# Dropping Duplicates
df = df.drop_duplicates()

# Balance Dataset
print(df['label'].value_counts())
phishing_emails = df[df['label'] == 'phishing']
non_phishing_emails = df[df['label'] == 'ham']

# Create New Dataset (5000 Phishing/5000 Non-Phishing)
phishing_sample = phishing_emails.sample(n=5000, random_state=42)
non_phishing_sample = non_phishing_emails.sample(n=5000, random_state=42)
df_sample = pd.concat([phishing_sample, non_phishing_sample]).reset_index(drop=True)
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# CSV
df_sample.to_csv('../parsed_datasets/balanced_phishing_dataset.csv', index=False)

rows, cols = df_sample.shape
print(f"The dataset contains {rows} rows and {cols} columns.")


def get_top_n_words(text, n=10):
    vectorizer = CountVectorizer(stop_words='english')
    word_count = vectorizer.fit_transform(text)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_count.sum(axis=0).A1))
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:n]

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()


phishing_text = df_sample[df_sample['label'] == 'phishing']['cleaned_body']
non_phishing_text = df_sample[df_sample['label'] == 'ham']['cleaned_body']
plot_wordcloud(phishing_text, "Most Common Words in Phishing Emails")
plot_wordcloud(non_phishing_text, "Most Common Words in Non-Phishing Emails")

# Top 10 Words
phishing_top_words = get_top_n_words(phishing_text, n=10)
non_phishing_top_words = get_top_n_words(non_phishing_text, n=10)

print("Top 10 Words in Phishing Emails:")
for word, freq in phishing_top_words:
    print(f"{word}: {freq}")

print("\nTop 10 Words in Non-Phishing Emails:")
for word, freq in non_phishing_top_words:
    print(f"{word}: {freq}")


# Pie Chart (Phishing v.s. Non-Phishing)
label_counts = df_sample['label'].value_counts()
labels = ['Non-Phishing', 'Phishing']
sizes = label_counts.values
colors = ['skyblue', 'lightcoral']
explode = (0.1, 0)

plt.figure(figsize=(8, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode
)
plt.title('Phishing vs. Non-Phishing Email Distribution')
plt.axis('equal')
plt.show()