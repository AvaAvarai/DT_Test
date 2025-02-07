import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to 2D (flatten 28x28 images into 784 features)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Min-Max Normalization
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Train a Decision Tree on the full dataset
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_normalized, y_train)

# Generate Graphviz DOT data
dot_data = export_graphviz(dt, out_file=None, filled=True, 
                           feature_names=[f'pixel_{i}' for i in range(X_train.shape[1])], 
                           class_names=[str(i) for i in range(10)], 
                           special_characters=True)

graph = graphviz.Source(dot_data)

# Save as PNG and PDF
graph.format = "png"
graph.render("mnist_decision_tree")
print("High-resolution PNG saved as 'mnist_decision_tree.png'.")


# Export full tree rules without truncation
tree_rules = export_text(dt, feature_names=[f'pixel_{i}' for i in range(X_train.shape[1])], max_depth=1000, show_weights=True)

with open("mnist_decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)

print("Decision tree rules saved as 'mnist_decision_tree_rules.txt'.")
