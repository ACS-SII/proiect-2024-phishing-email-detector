from keras.src.utils import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

df = pd.read_csv('../parsed_datasets/balanced_phishing_dataset.csv')

df['cleaned_body'] = df['cleaned_body'].fillna('').astype(str)
tokenized_sentences = [text.split() for text in df['cleaned_body']]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)

word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}

def text_to_sequence(text, word_index):
    return [word_index[word] for word in text.split() if word in word_index]

sequences = [text_to_sequence(text, word_index) for text in df['cleaned_body']]
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')  # Pad to max length

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['categorized_label'], test_size=0.2, random_state=42)

print("Word2Vec feature shape:", X_train.shape)


embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Model architecture
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=100),
    Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the model
model.save('./cnn_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cnn_predictions = model.predict(X_test)
cnn_predictions_binary = (cnn_predictions > 0.5).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(y_test, cnn_predictions_binary))
print("Classification Report:")
print(classification_report(y_test, cnn_predictions_binary))

# FOR PLOTTING
import matplotlib.pyplot as plt

# Extract data from history
epochs = range(1, len(history.history['accuracy']) + 1)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Phishing'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

