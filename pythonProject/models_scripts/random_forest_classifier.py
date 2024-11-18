from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from models_scripts.load_and_preprocess_data import load_and_preprocess_data

filepath = '../parsed_datasets/preprocessed_combined_phishing_dataset.csv'
X_train_tfidf, X_test_tfidf, y_train, y_test = load_and_preprocess_data(filepath, False)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
rf_model.fit(X_train_tfidf, y_train)
rf_predictions = rf_model.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Classifier Results")
print(f"Accuracy: {rf_accuracy}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))