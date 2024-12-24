import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

models = ['Logistic Regression', 'Random Forest', 'SVM']
accuracies = [0.683, 0.728, 0.665]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x, accuracies, width, label='Accuracy', color='skyblue')

plt.xticks(x, models, rotation=15)
plt.ylabel('Scores')
plt.title('Performance Metrics Comparison')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

confusion_matrices = {
    'Logistic Regression': [[85, 32], [38, 66]],
    'Random Forest': [[87, 30], [30, 74]],
    'SVM': [[94, 23], [51, 53]]
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
