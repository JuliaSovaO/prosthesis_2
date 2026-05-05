import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

confusion_matrix_data = np.array([
    [195,   0,   2,   0,   0,   0,   0,  33,   0,   0],  # finger-gun
    [  0, 208,   1,   0,  16,   0,   0,   0,   0,   2],  # four
    [  1,   1, 172,  13,   0,  11,   2,  10,   8,  14],  # fuck
    [  0,   3,   2, 231,   0,   0,   0,   0,   0,   0],  # good
    [  0,  42,   1,   1, 171,   0,   0,   0,   1,   8],  # okay
    [  0,   0,  15,   0,   0, 156,   0,  19,  23,  15],  # paper
    [  1,   0,   2,   0,   5,   0, 214,   0,   2,   2],  # rest
    [ 30,   0,  13,   1,   0,   8,   0, 183,   0,   1],  # rock
    [  1,   2,  17,   0,   1,  16,   0,   4, 178,  13],  # scissors
    [  1,   2,  23,   5,  13,  37,   1,   6,  24, 115],  # three
])

class_names = ['finger-gun', 'four', 'fuck', 'good', 'okay', 
               'paper', 'rest', 'rock', 'scissors', 'three']

per_class_accuracy = {
    'finger-gun': 84.8,
    'four': 91.6,
    'fuck': 74.1,
    'good': 97.9,
    'okay': 76.3,
    'paper': 68.4,
    'rest': 94.7,
    'rock': 77.5,
    'scissors': 76.7,
    'three': 50.7,
}

sample_counts = np.diag(confusion_matrix_data)
total_per_class = np.sum(confusion_matrix_data, axis=1)

# ============================================
# PLOT 1: Confusion Matrix (Heatmap)
# ============================================
fig, ax = plt.subplots(figsize=(14, 12))

cm_normalized = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={'label': 'Percentage (%)'})

ax.set_xlabel('Predicted Gesture', fontsize=16)
ax.set_ylabel('True Gesture', fontsize=16)
ax.set_title('Confusion Matrix - EMG Gesture Classification\n(Percentages, 79.3% Overall Accuracy)', fontsize=18, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/confusion_matrix.png")

# ============================================
# PLOT 2: Per-Class Accuracy Bar Chart
# ============================================
fig, ax = plt.subplots(figsize=(12, 6))

classes = list(per_class_accuracy.keys())
accuracies = list(per_class_accuracy.values())
colors = ['#2ecc71' if acc >= 80 else '#f39c12' if acc >= 70 else '#e74c3c' for acc in accuracies]

bars = ax.bar(classes, accuracies, color=colors, edgecolor='black', linewidth=0.5)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.axhline(y=79.3, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy: 79.3%')
ax.set_xlabel('Gesture', fontsize=16)
ax.set_ylabel('Accuracy (%)', fontsize=16)
ax.set_title('Per-Class Accuracy', fontsize=18, fontweight='bold')
ax.set_ylim(0, 105)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/per_class_accuracy.png")

# ============================================
# PLOT 3: Precision, Recall, F1-Score
# ============================================
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(cm):
    n_classes = cm.shape[0]
    precision = []
    recall = []
    f1 = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precision.append(prec * 100)
        recall.append(rec * 100)
        f1.append(f1_score * 100)
    
    return precision, recall, f1

precision, recall, f1 = calculate_metrics(confusion_matrix_data)

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', edgecolor='black')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Gesture', fontsize=16)
ax.set_ylabel('Score (%)', fontsize=16)
ax.set_title('Precision, Recall, and F1-Score by Gesture', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/precision_recall_f1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/precision_recall_f1.png")

# ============================================
# PLOT 4: Misclassification Analysis
# ============================================
fig, ax = plt.subplots(figsize=(12, 8))

misclassifications = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and confusion_matrix_data[i, j] > 0:
            misclassifications.append({
                'True': class_names[i],
                'Predicted': class_names[j],
                'Count': confusion_matrix_data[i, j]
            })

misclass_df = pd.DataFrame(misclassifications)
misclass_df = misclass_df.sort_values('Count', ascending=False).head(15)

bars = ax.barh(range(len(misclass_df)), misclass_df['Count'], color='#e67e22')
ax.set_yticks(range(len(misclass_df)))
ax.set_yticklabels([f"{row['True']} → {row['Predicted']}" for _, row in misclass_df.iterrows()])
ax.set_xlabel('Number of Misclassifications', fontsize=16)
ax.set_title('Top 15 Most Common Misclassifications', fontsize=18, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('plots/misclassifications.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/misclassifications.png")

# ============================================
# PRINT SUMMARY STATISTICS
# ============================================
print("\n" + "="*60)
print("CLASSIFICATION SUMMARY")
print("="*60)
print(f"Overall Accuracy: 79.3%")
print(f"Number of Classes: 10")
print(f"Test Samples: 2298")
print(f"Training Samples: 287,192")

print("\n" + "-"*60)
print("PER-CLASS PERFORMANCE")
print("-"*60)
print(f"{'Gesture':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<10}")
print("-"*60)
for i, name in enumerate(class_names):
    print(f"{name:<12} {per_class_accuracy[name]:<9.1f}% {precision[i]:<9.1f}% {recall[i]:<9.1f}% {f1[i]:<9.1f}% {total_per_class[i]:<10}")

print("\n" + "-"*60)
print("BEST PERFORMING GESTURES:")
print("-"*60)
for name, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"  ✓ {name}: {acc:.1f}%")

print("\n" + "-"*60)
print("WORST PERFORMING GESTURES:")
print("-"*60)
for name, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1])[:3]:
    print(f"  ✗ {name}: {acc:.1f}%")

print("\n" + "-"*60)
print("MOST COMMON CONFUSIONS:")
print("-"*60)
for _, row in misclass_df.head(5).iterrows():
    print(f"  {row['True']} → {row['Predicted']}: {row['Count']} times")

print("\nAll plots saved to 'plots/' directory")