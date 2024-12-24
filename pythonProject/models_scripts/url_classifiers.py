import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv('../parsed_datasets/urls_features_dataset.csv')
# print(df.iloc[1])
# Drop non-feature columns
X = df.drop(columns=["num_links", "extracted_links", "label"])
# Encode the target labels (0 for ham, 1 for phishing)
y = (df["label"] == "phishing").astype(int)
# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize scaler
scaler = StandardScaler()
# Fit on training data and transform (used for Logistic Regression and SVM)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_scaled = log_reg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_scaled))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_log_scaled))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_scaled))


# Initialize Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# Initialize SVM
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm_scaled = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm_scaled))
print("SVM Report:\n", classification_report(y_test, y_pred_svm_scaled))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_scaled))
