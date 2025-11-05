import numpy as np
import pandas as pd

trainingset = pd.read_csv('train.csv');
print(trainingset.head(20));
testset = pd.read_csv('test.csv');

def cleantrainingset(path= 'train.csv'):
    trainingdf = pd.read_csv('train.csv')
    medianage = trainingdf["Age"].median()
    trainingdf["Age"].fillna(medianage, inplace=True)
    trainingdf["Cabin"].fillna("U", inplace=True)
    trainingdf["CabinOnDeck"] = trainingdf["Cabin"].apply(lambda i : i[0] if isinstance(i, str) and i != "U" else "U")

    if trainingdf["Embarked"].isnull().any():
        kNN = 5;  
        commonport = trainingdf["Embarked"].mode().iloc[0]  ##iloc[0] mai pehli value is chosen, agar mode mai tie, it chooses pehli value from mode
        for i in trainingdf[trainingdf["Embarked"].isnull()].index:
            Pclass = trainingdf.at[i, "Pclass"]
            fare = trainingdf.at[i, "Fare"]
            potentialneighbours = trainingdf[(trainingdf["Pclass"] == Pclass) & (trainingdf["Fare"].notnull()) & (trainingdf["Embarked"].notnull())].copy()
            if potentialneighbours.empty:
                choice = commonport
            else:
                potentialneighbours["fareDifference"] = (potentialneighbours["Fare"] - fare).abs()
                nearestneighbour = potentialneighbours.nsmallest(kNN, "fareDifference")
                if not nearestneighbour.empty:
                    choice = nearestneighbour["Embarked"].mode().iloc[0]
                else:
                    choice = commonport
                trainingdf.at[i, "Embarked"] = choice             
    trainingdf["Embarked"].fillna(trainingdf["Embarked"].mode().iloc[0], inplace=True)  #if still missing values, replace them w most common port ki values

    trainingdf["Sex"] = trainingdf["Sex"].map({"male" : 0, "female" : 1}).astype(int)
    trainingdf["Embarked"] = trainingdf["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2}).astype(int)
    trainingdf["CabinOnDeck"] = pd.factorize(trainingdf["CabinOnDeck"])[0]

    cleanedfeatures = trainingdf[["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "CabinOnDeck", "Embarked"]].copy()
    cleanedfeatures.to_csv("cleanedtraincsv", index=False)
    return cleanedfeatures

def cleaningtestset(path= 'test.csv'):
    testdf = pd.read_csv('test.csv')
    testdf["Age"] = testdf["Age"].fillna(testdf["Age"].median())

    if 'Cabin' not in testdf.columns:   #incase cabins column wapis urrjae
        testdf["Cabin"] = "U"
    else:
        testdf["Cabin"].fillna("U")
    testdf["CabinOnDeck"] = testdf["Cabin"].apply(lambda i : i[0] if isinstance(i, str) and i != "U" else "U")

    if testdf["Embarked"].isnull().any():
        testdf["Embarked"] = testdf["Embarked"].fillna(testdf["Embarked"].mode().iloc[0])
    testdf["Fare"] = testdf["Fare"].fillna(testdf["Fare"].median())

    testdf["Sex"] = testdf["Sex"].map({"male" : 0, "female" : 1}).astype(int)
    testdf["Embarked"] = testdf["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2}).astype(int)
    testdf["CabinOnDeck"] = pd.factorize(testdf["CabinOnDeck"])[0]
    cleanedtestfeatures = testdf[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "CabinOnDeck", "Embarked"]].copy()
    cleanedtestfeatures.to_csv("cleanedtestcsv", index=False)
    return cleanedtestfeatures