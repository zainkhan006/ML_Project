import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('cleanedtrain.csv')

print("="*60)
print("TITANIC SURVIVAL PREDICTION - GINI & INFORMATION GAIN")
print("="*60)
print(f"\nDataset Shape: {df.shape}")
print(f"Target Variable: Survived (0=Died, 1=Survived)")
print("\nFirst 5 rows:")
print(df.head())

# ============================================
# MANUAL CALCULATION OF GINI INDEX
# ============================================
def calculate_gini(data, feature, target):
    """Calculate Gini Index for a feature"""
    total = len(data)
    gini_index = 0
    
    # Get unique values of the feature
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        
        if subset_size == 0:
            continue
        
        # Calculate Gini for this subset
        gini_subset = 1.0
        for label in data[target].unique():
            p = len(subset[subset[target] == label]) / subset_size
            gini_subset -= p ** 2
        
        # Weight by subset size
        gini_index += (subset_size / total) * gini_subset
    
    return gini_index

# ============================================
# MANUAL CALCULATION OF INFORMATION GAIN
# ============================================
def calculate_entropy(data, target):
    """Calculate entropy for target variable"""
    total = len(data)
    entropy = 0
    
    for label in data[target].unique():
        p = len(data[data[target] == label]) / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_information_gain(data, feature, target):
    """Calculate Information Gain for a feature"""
    total_entropy = calculate_entropy(data, target)
    total = len(data)
    weighted_entropy = 0
    
    # Calculate weighted entropy after split
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        subset_entropy = calculate_entropy(subset, target)
        weighted_entropy += (subset_size / total) * subset_entropy
    
    # Information Gain = Entropy(parent) - Weighted Entropy(children)
    information_gain = total_entropy - weighted_entropy
    
    return information_gain

# ============================================
# CALCULATE GINI & INFO GAIN FOR ALL FEATURES
# ============================================
print("\n" + "="*60)
print("CALCULATING GINI INDEX FOR ALL FEATURES")
print("="*60)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinOnDeck', 'Embarked']
target = 'Survived'

# Remove rows with missing values for calculation
df_clean = df.dropna()

gini_results = {}
for feature in features:
    gini = calculate_gini(df_clean, feature, target)
    gini_results[feature] = gini
    print(f"{feature:15s} | Gini Index: {gini:.4f}")

print("\n" + "="*60)
print("CALCULATING INFORMATION GAIN FOR ALL FEATURES")
print("="*60)

# Calculate parent entropy first
parent_entropy = calculate_entropy(df_clean, target)
print(f"\nParent Entropy (before any split): {parent_entropy:.4f}")
print("\nInformation Gain for each feature:")

ig_results = {}
for feature in features:
    ig = calculate_information_gain(df_clean, feature, target)
    ig_results[feature] = ig
    print(f"{feature:15s} | Information Gain: {ig:.4f}")

# ============================================
# SORT AND DISPLAY BEST FEATURES
# ============================================
print("\n" + "="*60)
print("FEATURE RANKING")
print("="*60)

print("\nBest Features by GINI INDEX (Lower is Better):")
sorted_gini = sorted(gini_results.items(), key=lambda x: x[1])
for i, (feature, gini) in enumerate(sorted_gini, 1):
    print(f"{i}. {feature:15s} | Gini: {gini:.4f}")

print("\nBest Features by INFORMATION GAIN (Higher is Better):")
sorted_ig = sorted(ig_results.items(), key=lambda x: x[1], reverse=True)
for i, (feature, ig) in enumerate(sorted_ig, 1):
    print(f"{i}. {feature:15s} | Info Gain: {ig:.4f}")

# ============================================
# TRAIN DECISION TREES USING BOTH CRITERIA
# ============================================
print("\n" + "="*60)
print("TRAINING DECISION TREES")
print("="*60)

# Prepare data
X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with Gini
print("\n1. Training Decision Tree with GINI INDEX...")
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
acc_gini = accuracy_score(y_test, y_pred_gini)
print(f"   Accuracy: {acc_gini:.4f} ({acc_gini*100:.2f}%)")

# Train with Entropy (Information Gain)
print("\n2. Training Decision Tree with ENTROPY (Information Gain)...")
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
acc_entropy = accuracy_score(y_test, y_pred_entropy)
print(f"   Accuracy: {acc_entropy:.4f} ({acc_entropy*100:.2f}%)")

# ============================================
# FEATURE IMPORTANCES FROM TRAINED TREES
# ============================================
print("\n" + "="*60)
print("FEATURE IMPORTANCES FROM TRAINED TREES")
print("="*60)

print("\nFeature Importances (Gini-based Tree):")
for feature, importance in zip(features, dt_gini.feature_importances_):
    print(f"{feature:15s} | Importance: {importance:.4f}")

print("\nFeature Importances (Entropy-based Tree):")
for feature, importance in zip(features, dt_entropy.feature_importances_):
    print(f"{feature:15s} | Importance: {importance:.4f}")

# ============================================
# VISUALIZE DECISION TREES
# ============================================
print("\n" + "="*60)
print("GENERATING DECISION TREE VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot Gini Tree
plot_tree(dt_gini, 
          feature_names=features,
          class_names=['Died', 'Survived'],
          filled=True,
          ax=axes[0],
          fontsize=8)
axes[0].set_title('Decision Tree with GINI INDEX', fontsize=14, fontweight='bold')

# Plot Entropy Tree
plot_tree(dt_entropy,
          feature_names=features,
          class_names=['Died', 'Survived'],
          filled=True,
          ax=axes[1],
          fontsize=8)
axes[1].set_title('Decision Tree with ENTROPY (Info Gain)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('decision_trees_comparison.png', dpi=300, bbox_inches='tight')
print("\n[OK] Decision trees saved as 'decision_trees_comparison.png'")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
[OK] Manual Gini Index calculated for all features
[OK] Manual Information Gain calculated for all features
[OK] Two Decision Trees trained:
  - CART with Gini Index: {acc_gini*100:.2f}% accuracy
  - CART with Entropy: {acc_entropy*100:.2f}% accuracy

KEY FINDINGS:
- Best feature by Gini: {sorted_gini[0][0]} (lowest Gini = {sorted_gini[0][1]:.4f})
- Best feature by Info Gain: {sorted_ig[0][0]} (highest IG = {sorted_ig[0][1]:.4f})

Both trees perform similarly, which is expected!
Decision trees visualized and saved.
""")

plt.show()