import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('cleanedtrain.csv')
print(f"\ndataset Shape: {df.shape}")
print(f"target variable: survived (0: died, 1: survived)")
print("\nfirst 5 rows:", df.head(5))

def calculate_gini(data, feature, target):
    total = len(data)
    gini_index = 0

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        
        if subset_size == 0:
            continue
        gini_subset = 1.0
        for label in data[target].unique():
            p = len(subset[subset[target] == label]) / subset_size
            gini_subset -= p ** 2
        
        gini_index += (subset_size / total) * gini_subset
    
    return gini_index

def calculate_entropy(data, target):
    total = len(data)
    entropy = 0
    
    for label in data[target].unique():
        p = len(data[data[target] == label]) / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_information_gain(data, feature, target):
    total_entropy = calculate_entropy(data, target)
    total = len(data)
    weighted_entropy = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        subset_entropy = calculate_entropy(subset, target)
        weighted_entropy += (subset_size / total) * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain


print("calculating gini: \n")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinOnDeck', 'Embarked']
target = 'Survived'
df_clean = df.dropna()
gini_results = {}
for feature in features:
    gini = calculate_gini(df_clean, feature, target)
    gini_results[feature] = gini
    print(f"{feature:15s} | gini index: {gini:.4f}")

print("calculating info gain: \n")
parent_entropy = calculate_entropy(df_clean, target)
print(f"\nparent entropy before any split: {parent_entropy:.4f}")
print("\ninformation gain for each feature:")

ig_results = {}
for feature in features:
    ig = calculate_information_gain(df_clean, feature, target)
    ig_results[feature] = ig
    print(f"{feature:15s} | information gain: {ig:.4f}")

print("feature ranking: \n")

print("best features by gini index:")  #lower is better
sorted_gini = sorted(gini_results.items(), key=lambda x: x[1])
for i, (feature, gini) in enumerate(sorted_gini, 1):
    print(f"{i}. {feature:15s} | Gini: {gini:.4f}")

print("\nbest Features by info gain: ") #higher is better
sorted_ig = sorted(ig_results.items(), key=lambda x: x[1], reverse=True)
for i, (feature, ig) in enumerate(sorted_ig, 1):
    print(f"{i}. {feature:15s} | Info Gain: {ig:.4f}")


print("\ntraining decision tree")

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\ntraining dt with gini index")
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
acc_gini = accuracy_score(y_test, y_pred_gini)
print(f"Accuracy: {acc_gini:.4f} ({acc_gini*100:.2f}%)")

print("\ntraining dt with Information gain")
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, y_pred_entropy)
print(f"Accuracy: {acc_entropy:.4f} ({acc_entropy*100:.2f}%)")

print("\nfeature importance: ")

print("\nfeature importances from gini based decision tree:")
for feature, importance in zip(features, dt_gini.feature_importances_):
    print(f"{feature:15s} | Importance: {importance:.4f}")

print("\nfeature importances from entropy based decision tree):")
for feature, importance in zip(features, dt_entropy.feature_importances_):
    print(f"{feature:15s} | Importance: {importance:.4f}")

print("\n visualisation of both decision trees: ")

plt.figure(figsize=(80, 40))
plot_tree(dt_gini, 
          feature_names=features,
          class_names=['Died', 'Survived'],
          filled=True,
          fontsize=6)
plt.title('decision tree w/ gini index', fontsize=14, fontweight='bold')
plt.savefig('decision_tree_gini.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(80, 40))
plot_tree(dt_entropy,
          feature_names=features,
          class_names=['Died', 'Survived'],
          filled=True,
          fontsize=6)
plt.title('decision tree w/ entropy(info gain)', fontsize=14, fontweight='bold')
plt.savefig('decision_tree_entropy.png', dpi=300, bbox_inches='tight')
print("\nDecision trees saved as 'decision_tree_gini.png' and 'decision_tree_entropy.png'")

print("\nSUMMARY: ")
print(f"""
Manual Gini index calculated for all features
Manual Information gain calculated for all features
Two Decision tres trained:
 1: CART with Gini: {acc_gini*100:.2f}% accuracy
 2: CART with info gain: {acc_entropy*100:.2f}% accuracy

key findigns:
- Best feature by Gini: {sorted_gini[0][0]} (lowest Gini = {sorted_gini[0][1]:.4f})
- Best feature by Info Gain: {sorted_ig[0][0]} (highest IG = {sorted_ig[0][1]:.4f})

Both trees perform similarly, with info gain having a slightly higher accuracy.
""")

plt.show()
