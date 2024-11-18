from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from models_scripts.load_and_preprocess_data import load_and_preprocess_data

filepath = '../parsed_datasets/preprocessed_combined_phishing_dataset.csv'
X_train_tfidf, X_test_tfidf, y_train, y_test = load_and_preprocess_data(filepath, True)

logreg = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
logreg.fit(X_train_tfidf, y_train)
logistic_predictions = logreg.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print("F1-Score:", f1_score(y_test, logistic_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))
print("Classification Report:")
print(classification_report(y_test, logistic_predictions))

