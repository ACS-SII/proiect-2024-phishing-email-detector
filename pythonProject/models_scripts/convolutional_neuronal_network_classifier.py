import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

preprocessed_combined_df = pd.read_csv('../parsed_datasets/preprocessed_combined_phishing_dataset.csv')
preprocessed_combined_df = preprocessed_combined_df[preprocessed_combined_df['body'].notna()]
preprocessed_combined_df['body'] = preprocessed_combined_df['body'].astype(str)
preprocessed_combined_df = preprocessed_combined_df[preprocessed_combined_df['body'].str.strip() != '']

X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_combined_df['body'],
    preprocessed_combined_df['label'],
    test_size=0.2,
    random_state=42
)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=200)
X_test_padded = pad_sequences(X_test_seq, maxlen=200)

cnn_model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

cnn_model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_padded, y_test, verbose=0)
cnn_predictions = cnn_model.predict(X_test_padded)
cnn_predictions_binary = (cnn_predictions > 0.5).astype(int)

print("Convolutional Neural Network Results")
print(f"Accuracy: {cnn_accuracy}")
print("Classification Report:")
print(classification_report(y_test, cnn_predictions_binary))

