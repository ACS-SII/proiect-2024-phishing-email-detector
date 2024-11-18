from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from models_scripts.load_and_preprocess_data import load_and_preprocess_data

filepath = '../parsed_datasets/preprocessed_combined_phishing_dataset.csv'
X_train_tfidf, X_test_tfidf, y_train, y_test = load_and_preprocess_data(filepath, True)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("Support Vector Machine Classifier Results")
print(f"Accuracy: {svm_accuracy}")
print("Classification Report:")
print(classification_report(y_test, svm_predictions))
