import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Open file picker to get .csv file
file_path = input("Enter the path to the .csv file: ")

# Parse columns, one is 'class' case-insensitive headered column, use as labels
df = pd.read_csv(file_path)
df['class'] = df['class'].str.lower()

# MinMax data to [0,1]
scaler = MinMaxScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Train an AdaBoost model
X = df[df.columns[:-1]]
y = df['class']
ab = AdaBoostClassifier(n_estimators=14, random_state=42)
ab.fit(X, y)

# Plot each individual decision stump
fig, axes = plt.subplots(nrows=ab.n_estimators, figsize=(10, 20))
for i, (ax, tree) in enumerate(zip(axes, ab.estimators_)):
    plot_tree(tree, ax=ax, feature_names=X.columns, class_names=y.unique(), filled=True)
    ax.set_title(f"Decision Stump {i+1}")
plt.tight_layout()
plt.show()

