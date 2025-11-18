import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cleanedtrain.csv')
print(f"dataset Shape: {df.shape}")  
print("row:", df.head(5))

#target variable is survived
#do not calculate gini or info gain(entropy) of Survived or PassengerID class
def calculateGini(data, feature, target):
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

def calculateEntropy(data, target):
    total = len(data)
    entropy = 0
    
    for label in data[target].unique():
        p = len(data[data[target] == label]) / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def calculateInfoGain(data, feature, target):
    total_entropy = calculateEntropy(data, target)
    total = len(data)
    weighted_entropy = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        subset_entropy = calculateEntropy(subset, target)
        weighted_entropy += (subset_size / total) * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain

############################################################ gini and info gain for original features ##########################################################

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinOnDeck', 'Embarked']
target = 'Survived'
df_clean = df.dropna()

print("calculating gini index:")
giniResults = {}
for feature in features:
    gini = calculateGini(df_clean, feature, target)
    giniResults[feature] = gini
    print(f"{feature:15s} | Gini Index: {gini:.4f}")

print("\ncalculating info gain:")
parentEntropy = calculateEntropy(df_clean, target)
print(f"parent Entropy before any split: {parentEntropy:.4f}")
print("info gain for each feature:")
igResults = {}
for feature in features:
    ig = calculateInfoGain(df_clean, feature, target)
    igResults[feature] = ig
    print(f"{feature:15s} | Information Gain: {ig:.4f}")

print("\nbest features sorted by gini:") #lower = :)
sortedGini = sorted(giniResults.items(), key=lambda x: x[1])
for i, (feature, gini) in enumerate(sortedGini, 1):
    print(f"{i}. {feature:15s} | Gini: {gini:.4f}")

print("\nbest features sorted by info gain:") #higher = :)
sortedIG = sorted(igResults.items(), key=lambda x: x[1], reverse=True)
for i, (feature, ig) in enumerate(sortedIG, 1):
    print(f"{i}. {feature:15s} | Info Gain: {ig:.4f}")

###################################################################feature engineering ####################################################################
print("\nfeature engineering:")

X = df[features].copy()
y = df[target]

X['AgeClass'] = X['Age'] * X['Pclass']
X['FarePerPerson'] = X['Fare'] / (X['SibSp'] + X['Parch'] + 1)
X['FarePerClass'] = X['Fare'] / X['Pclass']

print(f"Original features: {len(features)}")
print(f"Total features after engineering: {len(X.columns)}")
print(f"New features: {list(X.columns[len(features):])}")
print("feature engineering done :)")

############################################################splitting######################################################################################
print("\nsplitting data:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print("data splitted :)")

###########################################################scaling########################################################################################
print("\nscaling numerical features:")
X_train = X_train.copy()
X_test = X_test.copy()

scaler = StandardScaler()
numericalFeatures = ['Age', 'Fare', 'AgeClass', 'FarePerPerson', 'FarePerClass']
print(f"Scaling features: {numericalFeatures}")

X_train[numericalFeatures] = scaler.fit_transform(X_train[numericalFeatures])
X_test[numericalFeatures] = scaler.transform(X_test[numericalFeatures])

print("scaling done :)")

###########################################################training decision trees#######################################################################
print("\ntraining decision trees:")

print("training dt with gini index:")
dtGini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=2025)
dtGini.fit(X_train, y_train)
yPredGini = dtGini.predict(X_test)
GiniAccuracy = accuracy_score(y_test, yPredGini)
print(f"Accuracy: {GiniAccuracy:.4f} ({GiniAccuracy*100:.2f}%)")

print("training dt with info gain:")
dtEntropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dtEntropy.fit(X_train, y_train)
yPredEntropy = dtEntropy.predict(X_test)
EntropyAccuracy = accuracy_score(y_test, yPredEntropy)
print(f"Accuracy: {EntropyAccuracy:.4f} ({EntropyAccuracy*100:.2f}%)")

disagrees = (yPredGini != yPredEntropy).sum()
print(f"models disagree on {disagrees} out of {len(X_test)} predictions")

print("\nimportance of different features:")

print("\nfeature importance wrt gini:")
for feature, importance in zip(X.columns, dtGini.feature_importances_):
    if importance > 0:
        print(f"{feature:20s} | Importance: {importance:.4f}")

print("\nfeature importance wrt info gain:")
for feature, importance in zip(X.columns, dtEntropy.feature_importances_):
    if importance > 0:  
        print(f"{feature:20s} | Importance: {importance:.4f}")

print("dt trained :)")

#################################################################training random forest ###########################################################################
print("\ntraining random forest:")
gridParameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

baseRandomForest = RandomForestClassifier(random_state=2025)
gridSearch = GridSearchCV(
    estimator=baseRandomForest,
    param_grid=gridParameters,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

gridSearch.fit(X_train, y_train)
rfClassifier = gridSearch.best_estimator_
print("\nbest parameters:")
for param, value in gridSearch.best_params_.items():
    print(f"{param}: {value}")

CVScores = cross_val_score(rfClassifier, X_train, y_train, cv=5)
print(f"cross validation scores: {CVScores}")
print(f"avg cv score: {CVScores.mean():.4f} (+/- {CVScores.std() * 2:.4f})")

yPredRf = rfClassifier.predict(X_test)
RFAccuracy = accuracy_score(y_test, yPredRf)
print(f"random forest test accuracy: {RFAccuracy:.4f} ({RFAccuracy*100:.2f}%)")

print("\nrandom forest most important features:")
featureImportances = sorted(zip(X.columns, rfClassifier.feature_importances_), key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(featureImportances[:10], 1):
    print(f"{i:2d}. {feature:20s} | Importance: {importance:.4f}")

disagreeRfAndGini = (yPredRf != yPredGini).sum()
disagreeRfAndEntropy = (yPredRf != yPredEntropy).sum()
print(f"random forest disagrees with gini index on {disagreeRfAndGini} out of {len(X_test)} predictions")
print(f"random forest disagrees with entropy on {disagreeRfAndEntropy} out of {len(X_test)} predictions")
print("random forest trained :)")

##################################################################### fin ##################################################################################
print("\nmaking predictions on testing data:")

test = pd.read_csv('cleanedtest.csv')
print(f"test set shape: {test.shape}")

testFeatures = test[features].copy()
testFeatures['AgeClass'] = testFeatures['Age'] * testFeatures['Pclass']
testFeatures['FarePerPerson'] = testFeatures['Fare'] / (testFeatures['SibSp'] + testFeatures['Parch'] + 1)
testFeatures['FarePerClass'] = testFeatures['Fare'] / testFeatures['Pclass']
testFeatures[numericalFeatures] = scaler.transform(testFeatures[numericalFeatures])

dtPredictions = dtEntropy.predict(testFeatures)
dtSubmission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': dtPredictions
})
dtSubmission.to_csv('dtpredictions.csv', index=False)
print("dt file saved")

rfPredictions = rfClassifier.predict(testFeatures)
rfSubmission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': rfPredictions
})
rfSubmission.to_csv('rfpredictions.csv', index=False)
print("rf file saved")
print(rfSubmission.head(10))

































































































































































































































































# print("calculating gini: \n")

# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinOnDeck', 'Embarked']
# target = 'Survived'
# df_clean = df.dropna()
# gini_results = {}
# for feature in features:
#     gini = calculate_gini(df_clean, feature, target)
#     gini_results[feature] = gini
#     print(f"{feature:15s} | gini index: {gini:.4f}")

# print("calculating info gain: \n")
# parent_entropy = calculate_entropy(df_clean, target)
# print(f"\nparent entropy before any split: {parent_entropy:.4f}")
# print("\ninformation gain for each feature:")

# ig_results = {}
# for feature in features:
#     ig = calculate_information_gain(df_clean, feature, target)
#     ig_results[feature] = ig
#     print(f"{feature:15s} | information gain: {ig:.4f}")

# print("feature ranking: \n")

# print("best features by gini index:")  #lower is better
# sorted_gini = sorted(gini_results.items(), key=lambda x: x[1])
# for i, (feature, gini) in enumerate(sorted_gini, 1):
#     print(f"{i}. {feature:15s} | Gini: {gini:.4f}")

# print("\nbest Features by info gain: ") #higher is better
# sorted_ig = sorted(ig_results.items(), key=lambda x: x[1], reverse=True)
# for i, (feature, ig) in enumerate(sorted_ig, 1):
#     print(f"{i}. {feature:15s} | Info Gain: {ig:.4f}")


# print("\ntraining decision tree")

# X = df[features].copy()
# y = df[target]
# X['Age_Class'] = X['Age'] * X['Pclass']
# X['Fare_Per_Person'] = X['Fare'] / (X['SibSp'] + X['Parch'] + 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("\ntraining dt with gini index")
# dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
# dt_gini.fit(X_train, y_train)
# y_pred_gini = dt_gini.predict(X_test)
# acc_gini = accuracy_score(y_test, y_pred_gini)
# print(f"Accuracy: {acc_gini:.4f} ({acc_gini*100:.2f}%)")

# print("\ntraining dt with Information gain")
# dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
# dt_entropy.fit(X_train, y_train)
# y_pred_entropy = dt_entropy.predict(X_test)
# acc_entropy = accuracy_score(y_test, y_pred_entropy)
# print(f"Accuracy: {acc_entropy:.4f} ({acc_entropy*100:.2f}%)")

# print("\nfeature importance: ")

# print("\nfeature importances from gini based decision tree:")
# for feature, importance in zip(features, dt_gini.feature_importances_):
#     print(f"{feature:15s} | Importance: {importance:.4f}")

# print("\nfeature importances from entropy based decision tree):")
# for feature, importance in zip(features, dt_entropy.feature_importances_):
#     print(f"{feature:15s} | Importance: {importance:.4f}")

# # print("\n visualisation of both decision trees: ")

# # plt.figure(figsize=(80, 40))
# # plot_tree(dt_gini, 
# #           feature_names=features,
# #           class_names=['Died', 'Survived'],
# #           filled=True,
# #           fontsize=6)
# # plt.title('decision tree w/ gini index', fontsize=14, fontweight='bold')
# # plt.savefig('decision_tree_gini.png', dpi=300, bbox_inches='tight')

# # plt.figure(figsize=(80, 40))
# # plot_tree(dt_entropy,
# #           feature_names=features,
# #           class_names=['Died', 'Survived'],
# #           filled=True,
# #           fontsize=6)
# # plt.title('decision tree w/ entropy(info gain)', fontsize=14, fontweight='bold')
# # plt.savefig('decision_tree_entropy.png', dpi=300, bbox_inches='tight')
# # print("\nDecision trees saved as 'decision_tree_gini.png' and 'decision_tree_entropy.png'")

# print("\nSUMMARY: ")
# print(f"""
# Manual Gini index calculated for all features
# Manual Information gain calculated for all features
# Two Decision trees trained:
#  1: CART with Gini: {acc_gini*100:.2f}% accuracy
#  2: CART with info gain: {acc_entropy*100:.2f}% accuracy

# Key findings:
# - Best feature by Gini: {sorted_gini[0][0]} (lowest Gini = {sorted_gini[0][1]:.4f})
# - Best feature by Info Gain: {sorted_ig[0][0]} (highest IG = {sorted_ig[0][1]:.4f})

# Both trees perform similarly, with info gain having a slightly higher accuracy.
# """)

# plt.show()

# #seeing how many predictions are more or less same
# giniprediction = dt_gini.predict(X_test)
# entropyprediction = dt_entropy.predict(X_test)
# disagree = (giniprediction != entropyprediction).sum()
# print(f"model disagrees on {disagree} out of {len(X_test)} predictions")

# # total = 891
# # 80% of 891 = 718
# # so training = 718 features, validation set = 179 features
# # model disagrees on 2 out of 179 features, hence we will use entropy wali tree to make predictions since almost all predictions bw gini and entropy tree are same

# test = pd.read_csv('cleanedtest.csv')
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinOnDeck', 'Embarked']
# xtest_entropy = test[features].copy()
# xtest_entropy['Age_Class'] = xtest_entropy['Age'] * xtest_entropy['Pclass']
# xtest_entropy['Fare_Per_Person'] = xtest_entropy['Fare'] / (xtest_entropy['SibSp'] + xtest_entropy['Parch'] + 1)
# xtest_entropy['Fare Per Class'] = xtest_entropy['Fare'] / xtest_entropy['Pclass']
# predictions = dt_entropy.predict(xtest_entropy)
# predictionfile = pd.DataFrame({
#     'PassengerId': test['PassengerId'],
#     'Survived': predictions
# })
# predictionfile.to_csv('predictions.csv', index=False)
# print(f"first few predictions: \n{predictionfile.head(10)}")


# #Random Forest Classifier with Optimization
# print("\n\nOptimizing Random Forest Classifier...")

# X_train = X_train.copy()
# X_test = X_test.copy()

# # Feature Engineering
# X_train['Age_Class'] = X_train['Age'] * X_train['Pclass']
# X_train['Fare_Per_Person'] = X_train['Fare'] / (X_train['SibSp'] + X_train['Parch'] + 1)
# X_train['Fare Per Class'] = X_train['Fare'] / X_train['Pclass']
# X_test['Age_Class'] = X_test['Age'] * X_test['Pclass']
# X_test['Fare_Per_Person'] = X_test['Fare'] / (X_test['SibSp'] + X_test['Parch'] + 1)
# X_test['Fare Per Class'] = X_test['Fare'] / X_test['Pclass']

# # Scale numerical features
# scaler = StandardScaler()
# numerical_features = ['Age', 'Fare', 'Age_Class', 'Fare_Per_Person', 'Fare Per Class']
# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# # Define parameter grid for GridSearchCV
# param_grid = {
#     'n_estimators': [200, 300],
#     'max_depth': [5, 7],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'max_features': ['sqrt', 'log2'],
#     'class_weight': ['balanced', None]
# }

# # Initialize base Random Forest
# base_rf = RandomForestClassifier(random_state=42)

# # Perform Grid Search with Cross Validation
# print("Performing Grid Search with 5-fold Cross Validation...")
# grid_search = GridSearchCV(
#     estimator=base_rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )

# # Fit the grid search
# grid_search.fit(X_train, y_train)

# # Get best model
# rf_classifier = grid_search.best_estimator_

# print(f"\nBest parameters found:")
# for param, value in grid_search.best_params_.items():
#     print(f"{param}: {value}")

# # Perform cross-validation on the best model
# cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
# print(f"\nCross-validation scores: {cv_scores}")
# print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# # Train the final model with best parameters
# rf_classifier.fit(X_train, y_train)

# # Make predictions
# y_pred_rf = rf_classifier.predict(X_test)
# acc_rf = accuracy_score(y_test, y_pred_rf)
# print(f"\nRandom Forest Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

# # Compare feature importances
# print("\nFeature importances from Random Forest:")
# for feature, importance in zip(features, rf_classifier.feature_importances_):
#     print(f"{feature:15s} | Importance: {importance:.4f}")

# # Compare predictions with Decision Trees
# rf_predictions = rf_classifier.predict(X_test)
# disagree_rf_gini = (rf_predictions != giniprediction).sum()
# disagree_rf_entropy = (rf_predictions != entropyprediction).sum()
# print(f"\nRandom Forest disagrees with Gini tree on {disagree_rf_gini} out of {len(X_test)} predictions")
# print(f"Random Forest disagrees with Entropy tree on {disagree_rf_entropy} out of {len(X_test)} predictions")

# # Add engineered features to test set
# xtest_entropy['Age_Class'] = xtest_entropy['Age'] * xtest_entropy['Pclass']
# xtest_entropy['Fare_Per_Person'] = xtest_entropy['Fare'] / (xtest_entropy['SibSp'] + xtest_entropy['Parch'] + 1)

# # Scale numerical features for test set
# xtest_entropy[numerical_features] = scaler.transform(xtest_entropy[numerical_features])

# # Make predictions on test set using optimized Random Forest
# rf_test_predictions = rf_classifier.predict(xtest_entropy)
# rf_predictionfile = pd.DataFrame({
#     'PassengerId': test['PassengerId'],
#     'Survived': rf_test_predictions
# })
# rf_predictionfile.to_csv('rf_predictions.csv', index=False)
# print(f"\nRandom Forest predictions saved to 'rf_predictions.csv'")
# print(f"First few Random Forest predictions:\n{rf_predictionfile.head(10)}")

# # print("\nFINAL SUMMARY:")
# # print(f"""
# # Model Comparison:
# # 1. Decision Tree (Gini): {acc_gini*100:.2f}% accuracy
# # 2. Decision Tree (Info Gain): {acc_entropy*100:.2f}% accuracy
# # 3. Random Forest: {acc_rf*100:.2f}% accuracy

# # The Random Forest classifier combines multiple decision trees to make predictions.
# # Each tree in the forest is trained on a different subset of the data, which helps
# # reduce overfitting and generally leads to more robust predictions than a single
# # decision tree.""")