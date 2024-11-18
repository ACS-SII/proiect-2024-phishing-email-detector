import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

preprocessed_combined_df = pd.read_csv('./parsed_datasets/preprocessed_combined_phishing_dataset.csv')
label_counts = preprocessed_combined_df['label'].value_counts()
print("Label Counts:")
print(label_counts)

ham_count = label_counts.get('ham', 0)
phishing_count = label_counts.get('phishing', 0)

print(f"\nNumber of 'ham' emails: {ham_count}")
print(f"Number of 'phishing' emails: {phishing_count}")


plt.bar(['Ham', 'Phishing'], [ham_count, phishing_count])
plt.ylabel('Number of Emails')
plt.title('Ham vs Phishing Emails')
plt.show()


preprocessed_combined_df = preprocessed_combined_df[preprocessed_combined_df['body'].notna()]
preprocessed_combined_df['body'] = preprocessed_combined_df['body'].astype(str)
preprocessed_combined_df = preprocessed_combined_df[preprocessed_combined_df['body'].str.strip() != '']

phishing_text = ' '.join(preprocessed_combined_df[preprocessed_combined_df['label'] == 'phishing']['body'])
ham_text = ' '.join(preprocessed_combined_df[preprocessed_combined_df['label'] == 'ham']['body'])

stop_words = stopwords.words('english')
stop_words.extend(["nbsp", "font", "sans", "serif", "bold", "arial", "verdana", "helvetica", "http", "https", "www", "html", "enron", "margin", "spamassassin"])

wordcloudPhishing = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words).generate(phishing_text)

print(wordcloudPhishing.words_.keys())

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloudPhishing, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Phishing Emails')
plt.tight_layout(pad=0)
plt.show()
