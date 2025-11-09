import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('cleanedtrain.csv')
print(f"\ndataset Shape: {df.shape}")
print(f"target variable: survived (0: died, 1: survived)")
print("\n row: ", df.head(5))

#do not calculate gini or info gain(entropy) of Survived or PassengerID class
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
Two Decision trees trained:
 1: CART with Gini: {acc_gini*100:.2f}% accuracy
 2: CART with info gain: {acc_entropy*100:.2f}% accuracy

Key findings:
- Best feature by Gini: {sorted_gini[0][0]} (lowest Gini = {sorted_gini[0][1]:.4f})
- Best feature by Info Gain: {sorted_ig[0][0]} (highest IG = {sorted_ig[0][1]:.4f})

Both trees perform similarly, with info gain having a slightly higher accuracy.
""")

plt.show()

#seeing how many predictions are more or less same
giniprediction = dt_gini.predict(X_test)
entropyprediction = dt_entropy.predict(X_test)
disagree = (giniprediction != entropyprediction).sum()
print(f"model disagrees on {disagree} out of {len(X_test)} predictions")

# total = 891
# 80% of 891 = 718
# so training = 718 features, validation set = 179 features
# model disagrees on 2 out of 179 features, hence we will use entropy wali tree to make predictions since almost all predictions bw gini and entropy tree are same

test = pd.read_csv('cleanedtest.csv')
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinOnDeck', 'Embarked']
xtest_entropy = test[features]
predictions = dt_entropy.predict(xtest_entropy)
predictionfile = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
predictionfile.to_csv('predictions.csv', index=False)
print(f"first few predictions: \n{predictionfile.head(10)}")


#Random Forrest Classifier
print("\n\nTraining Random Forest Classifier...")

# Initialize Random Forest with 100 trees
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

# Compare feature importances
print("\nFeature importances from Random Forest:")
for feature, importance in zip(features, rf_classifier.feature_importances_):
    print(f"{feature:15s} | Importance: {importance:.4f}")

# Compare predictions with Decision Trees
rf_predictions = rf_classifier.predict(X_test)
disagree_rf_gini = (rf_predictions != giniprediction).sum()
disagree_rf_entropy = (rf_predictions != entropyprediction).sum()
print(f"\nRandom Forest disagrees with Gini tree on {disagree_rf_gini} out of {len(X_test)} predictions")
print(f"Random Forest disagrees with Entropy tree on {disagree_rf_entropy} out of {len(X_test)} predictions")

# Make predictions on test set using Random Forest
rf_test_predictions = rf_classifier.predict(xtest_entropy)
rf_predictionfile = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': rf_test_predictions
})
rf_predictionfile.to_csv('rf_predictions.csv', index=False)
print(f"\nRandom Forest predictions saved to 'rf_predictions.csv'")
print(f"First few Random Forest predictions:\n{rf_predictionfile.head(10)}")

print("\nFINAL SUMMARY:")
print(f"""
Model Comparison:
1. Decision Tree (Gini): {acc_gini*100:.2f}% accuracy
2. Decision Tree (Info Gain): {acc_entropy*100:.2f}% accuracy
3. Random Forest: {acc_rf*100:.2f}% accuracy

The Random Forest classifier combines multiple decision trees to make predictions.
Each tree in the forest is trained on a different subset of the data, which helps
reduce overfitting and generally leads to more robust predictions than a single
decision tree.""")