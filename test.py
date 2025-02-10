from collections import Counter
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser="auto")
X, y = mnist.data, mnist.target

print(f"Total number of training cases: {len(X)}")

# Count cases per class in training data
class_counts = Counter(y)
print("\nCases per class in training data:")
for class_label, count in sorted(class_counts.items()):
    print(f"Class {class_label}: {count} cases")

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Train Decision Tree on full dataset with no depth limit
dt = DecisionTreeClassifier(random_state=42, max_depth=None)
dt.fit(X_normalized, y)

# Get leaf indices for each sample
leaf_indices = dt.apply(X_normalized)

# Count number of cases per terminal node and get majority class
leaf_counts = Counter(leaf_indices)
leaf_classes = {}
for leaf_id in leaf_counts:
    # Get samples in this leaf
    leaf_mask = leaf_indices == leaf_id
    # Get their classes
    leaf_y = y[leaf_mask]
    # Get majority class
    majority_class = Counter(leaf_y).most_common(1)[0][0]
    leaf_classes[leaf_id] = majority_class

# Count number of nodes per class
nodes_per_class = Counter(leaf_classes.values())
print("\nNumber of terminal nodes per class:")
for class_label, node_count in sorted(nodes_per_class.items()):
    print(f"Class {class_label}: {node_count} nodes")

# Write results to CSV file
with open('decision_tree_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Total terminal nodes', dt.get_n_leaves()])
    writer.writerow(['Node ID', 'Cases', 'Majority Class'])
    # Sort leaf_counts by count in descending order
    sorted_nodes = sorted(leaf_counts.items(), key=lambda x: x[1], reverse=True)
    for node, count in sorted_nodes:
        writer.writerow([node, count, leaf_classes[node]])