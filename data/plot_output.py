import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# Set global font sizes for all plots
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 18

confusion_matrix_data = np.array([
    [110,   0,   0,   1,   0,   0,   1,   0,   0,   0],  # finger-gun
    [  0, 101,   4,   0,   0,   5,   0,   0,   2,   0],  # four
    [  0,   8,  78,   0,   0,  12,   0,   7,   2,   2],  # fuck
    [  0,   0,   0, 110,   0,   0,   0,   0,   1,   0],  # good
    [  0,   0,   0,   0, 110,   0,   0,   0,   0,   2],  # okay
    [  1,  22,   6,   0,   0, 107,   0,   8,   4,   1],  # paper
    [  1,   0,   0,   0,   0,   0, 107,   0,   0,   0],  # rest
    [  0,   1,   3,   0,   0,   7,   0,  83,   2,   2],  # rock
    [  1,   4,   8,   1,   0,  12,   0,   4,  84,   1],  # scissors
    [  0,   0,   1,   0,  15,   3,   1,   6,   2,  84],  # three
])

class_names = ['finger-gun', 'four', 'fuck', 'good', 'okay', 
               'paper', 'rest', 'rock', 'scissors', 'three']

per_class_accuracy = {
    'finger-gun': 98.21,
    'four': 90.18,
    'fuck': 71.56,
    'good': 99.10,
    'okay': 98.21,
    'paper': 71.81,
    'rest': 99.07,
    'rock': 84.69,
    'scissors': 73.04,
    'three': 75.00,
}

sample_counts = np.diag(confusion_matrix_data)
total_per_class = np.sum(confusion_matrix_data, axis=1)

# PLOT 1: Confusion Matrix (Heatmap)
fig, ax = plt.subplots(figsize=(18, 16))

cm_normalized = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis] * 100

# Create heatmap with VERY LARGE annotation font size
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
            annot_kws={'size': 20, 'weight': 'bold'})  # Huge numbers inside matrix

ax.set_xlabel('Predicted Gesture', fontsize=26, fontweight='bold', labelpad=15)
ax.set_ylabel('True Gesture', fontsize=26, fontweight='bold', labelpad=15)
ax.set_title('Confusion Matrix - EMG Gesture Classification\n(Percentages, 85.6% Overall Accuracy)', 
             fontsize=32, fontweight='bold', pad=25)

# Increase tick label sizes
plt.xticks(rotation=45, ha='right', fontsize=22)
plt.yticks(rotation=0, fontsize=22)

# Increase colorbar label and tick size
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Percentage (%)', fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/confusion_matrix.png")

# PLOT 2: Per-Class Accuracy Bar Chart
fig, ax = plt.subplots(figsize=(16, 10))

classes = list(per_class_accuracy.keys())
accuracies = list(per_class_accuracy.values())
colors = ['#2ecc71' if acc >= 80 else '#f39c12' if acc >= 70 else '#e74c3c' for acc in accuracies]

bars = ax.bar(classes, accuracies, color=colors, edgecolor='black', linewidth=2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=20, fontweight='bold')

ax.axhline(y=85.6, color='red', linestyle='--', linewidth=3, label=f'Overall Accuracy: 85.6%')
ax.set_xlabel('Gesture', fontsize=26, fontweight='bold', labelpad=15)
ax.set_ylabel('Accuracy (%)', fontsize=26, fontweight='bold', labelpad=15)
ax.set_title('Per-Class Accuracy', fontsize=32, fontweight='bold', pad=25)
ax.set_ylim(0, 105)
ax.legend(loc='upper right', fontsize=18, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)

plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('plots/per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/per_class_accuracy.png")

# PLOT 3: Precision, Recall, F1-Score
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

fig, ax = plt.subplots(figsize=(18, 10))
x = np.arange(len(class_names))
width = 0.28

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black', linewidth=2)
bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', edgecolor='black', linewidth=2)
bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', edgecolor='black', linewidth=2)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 1.5, 
                   f'{height:.0f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.set_xlabel('Gesture', fontsize=26, fontweight='bold', labelpad=15)
ax.set_ylabel('Score (%)', fontsize=26, fontweight='bold', labelpad=15)
ax.set_title('Precision, Recall, and F1-Score by Gesture', fontsize=28, fontweight='bold', pad=25)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=20)
ax.legend(loc='upper right', fontsize=20, frameon=True, fancybox=True, shadow=True)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
ax.tick_params(axis='both', labelsize=20, width=2)

plt.tight_layout()
plt.savefig('plots/precision_recall_f1.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/precision_recall_f1.png")

# PLOT 4: Misclassification Analysis
fig, ax = plt.subplots(figsize=(16, 12))

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

bars = ax.barh(range(len(misclass_df)), misclass_df['Count'], color='#e67e22', edgecolor='black', linewidth=2, height=0.7)

# Add value labels
for i, (_, row) in enumerate(misclass_df.iterrows()):
    ax.text(row['Count'] + 1, i, str(row['Count']), va='center', fontsize=18, fontweight='bold')

ax.set_yticks(range(len(misclass_df)))
ax.set_yticklabels([f"{row['True']} → {row['Predicted']}" for _, row in misclass_df.iterrows()], fontsize=20)
ax.set_xlabel('Number of Misclassifications', fontsize=26, fontweight='bold', labelpad=15)
ax.set_title('Top 15 Most Common Misclassifications', fontsize=28, fontweight='bold', pad=25)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x', linewidth=1.5)
ax.tick_params(axis='both', labelsize=18, width=2)

plt.tight_layout()
plt.savefig('plots/misclassifications.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/misclassifications.png")

# PRINT SUMMARY STATISTICS
print("\n" + "="*70)
print("CLASSIFICATION SUMMARY")
print("="*70)
print(f"Overall Accuracy: 85.6%")
print(f"Number of Classes: 10")
print(f"Test Samples: 2298")
print(f"Training Samples: 287,192")

print("\n" + "-"*70)
print("PER-CLASS PERFORMANCE")
print("-"*70)
print(f"{'Gesture':<14} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
print("-"*70)
for i, name in enumerate(class_names):
    print(f"{name:<14} {per_class_accuracy[name]:<11.1f}% {precision[i]:<11.1f}% {recall[i]:<11.1f}% {f1[i]:<11.1f}% {total_per_class[i]:<10}")

print("\n" + "-"*70)
print("BEST PERFORMING GESTURES:")
print("-"*70)
for name, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"  ✓ {name}: {acc:.1f}%")

print("\n" + "-"*70)
print("WORST PERFORMING GESTURES:")
print("-"*70)
for name, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1])[:3]:
    print(f"  ✗ {name}: {acc:.1f}%")

print("\n" + "-"*70)
print("MOST COMMON CONFUSIONS:")
print("-"*70)
for _, row in misclass_df.head(5).iterrows():
    print(f"  {row['True']} → {row['Predicted']}: {row['Count']} times")

print("\nAll plots saved to 'plots/' directory")