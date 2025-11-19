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

###############################################################encoding##################################################################################
print("\n encoding categorical values")
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"sex before encoding: {df['Sex'].unique()}")
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
print(f"sex after encoding: {df['Sex'].unique()}")

print(f"embarked before encoding: {df['Embarked'].unique()}")
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
print(f"embarked after encoding: {df['Embarked'].unique()}")

print(f"deck before encoding: {df['Deck'].unique()}")
df['Deck'] = df['Deck'].map({'M': 0, 'ABC': 1, 'DE': 2, 'FG': 3}).astype(int)
print(f"deck after encoding: {df['Deck'].unique()}")

#################################################3gini and entropy functions###########################################################################

# target variable is survived
# do not calculate gini or info gain(entropy) of Survived or PassengerID class
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

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Deck', 'Embarked']
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

if 'Unnamed: 0' in test.columns:
    test = test.drop('Unnamed: 0', axis=1)

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).astype(int)
test['Embarked'] = test['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
test['Deck'] = test['Deck'].map({'M': 0, 'ABC': 1, 'DE': 2, 'FG': 3}).astype(int)

testFeatures = test[features].copy()
testFeatures['AgeClass'] = testFeatures['Age'] * testFeatures['Pclass']
testFeatures['FarePerPerson'] = testFeatures['Fare'] / (testFeatures['SibSp'] + testFeatures['Parch'] + 1)
testFeatures['FarePerClass'] = testFeatures['Fare'] / testFeatures['Pclass']
testFeatures[numericalFeatures] = scaler.transform(testFeatures[numericalFeatures])

dtPredictions = dtEntropy.predict(testFeatures)
dtSubmission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': dtPredictions.astype(int)
})
dtSubmission.to_csv('dtpredictions.csv', index=False)
print("dt file saved")

rfPredictions = rfClassifier.predict(testFeatures)
rfSubmission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': rfPredictions.astype(int)
})
rfSubmission.to_csv('rfpredictions.csv', index=False)
print("rf file saved")
print(rfSubmission.head(10))

##########################################visualisations##################################################################
# print("\n visualisation of both decision trees: ")

# plt.figure(figsize=(80, 40))
# plot_tree(dtGini, 
#           feature_names=features,
#           class_names=['Died', 'Survived'],
#           filled=True,
#           fontsize=6)
# plt.title('decision tree w/ gini index', fontsize=14, fontweight='bold')
# plt.savefig('decision_tree_gini.png', dpi=300, bbox_inches='tight')

# plt.figure(figsize=(80, 40))
# plot_tree(dtEntropy,
#           feature_names=features,
#           class_names=['Died', 'Survived'],
#           filled=True,
#           fontsize=6)
# plt.title('decision tree w/ entropy(info gain)', fontsize=14, fontweight='bold')
# plt.savefig('decision_tree_entropy.png', dpi=300, bbox_inches='tight')
# print("\nDecision trees saved as 'decision_tree_gini.png' and 'decision_tree_entropy.png'")
