import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

models = ['Logistic Regression', 'Random Forest', 'SVM', 'CNN']
accuracies = [0.984, 0.9795, 0.9799, 0.9894]
f1_scores = [0.868, 0.81, 0.82, 0.91]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, accuracies, width, label='Accuracy', color='skyblue')
plt.bar(x, f1_scores, width, label='F1-Score', color='blue')

plt.xticks(x, models, rotation=15)
plt.ylabel('Scores')
plt.title('Performance Metrics Comparison')
plt.ylim(0.75, 1)
plt.legend()
plt.tight_layout()
plt.show()

confusion_matrices = {
    'Logistic Regression': [[18522, 145], [173, 1048]],
    'Random Forest': [[18667, 0], [342, 879]],
    'SVM': [[18667, 0], [293, 928]],
    'CNN': [[18667, 0], [122, 1099]]
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (model, matrix) in enumerate(confusion_matrices.items()):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(model)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

epochs = list(range(1, 11))
train_accuracy = [0.9764, 0.9928, 0.9952, 0.9957, 0.9966, 0.9971, 0.9968, 0.9973, 0.9970, 0.9972]
val_accuracy = [0.9908, 0.9899, 0.9884, 0.9901, 0.9889, 0.9899, 0.9877, 0.9884, 0.9894, 0.9888]
train_loss = [0.0853, 0.0213, 0.0128, 0.0097, 0.0079, 0.0066, 0.0058, 0.0062, 0.0054, 0.0053]
val_loss = [0.0313, 0.0331, 0.0518, 0.0529, 0.0918, 0.0762, 0.1402, 0.0578, 0.0825, 0.1010]

# Graph 1: Comparing Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, marker='x', label='Train Loss', color='red')
plt.plot(epochs, val_loss, marker='x', label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss over 10 Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Comparing Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, marker='o', label='Train Accuracy', color='blue')
plt.plot(epochs, val_accuracy, marker='o', label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy over 10 Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


