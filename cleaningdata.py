import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

def concatenateDf(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divideDf(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

dfTrain = pd.read_csv('train.csv')
print(dfTrain.head(20))
dfTest = pd.read_csv('test.csv')
print("\n", dfTest.head(20))
dfAll = concatenateDf(dfTrain, dfTest)

dfTrain.name = 'Training Set'
dfTest.name = 'Test Set'
dfAll.name = 'Combined Set' 

dfs = [dfTrain, dfTest]

print('\nNumber of Training Examples = {}'.format(dfTrain.shape[0]))
print('Number of Test Examples = {}\n'.format(dfTest.shape[0]))
print('Training X Shape = {}'.format(dfTrain.shape))
print('Training y Shape = {}\n'.format(dfTrain['Survived'].shape[0]))
print('Test X Shape = {}'.format(dfTest.shape))
print('Test y Shape = {}\n'.format(dfTest.shape[0]))
print(dfTrain.columns)
print(dfTest.columns)

print(dfTrain.info())
print(dfTrain.sample(3))
print(dfTest.info())
print(dfTest.sample(3))

def displayMissing(df):    
    for col in df.columns.tolist():          
        print('\n {} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')
    
for df in dfs:
    print('{}'.format(df.name))
    displayMissing(df)

dfAllCorrelations = dfAll.corr(numeric_only=True).abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
dfAllCorrelations.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
dfAllCorrelations[dfAllCorrelations['Feature 1'] == 'Age']
print(dfAllCorrelations)

AgeByPclassAndSex = dfAll.groupby(['Sex', 'Pclass'])['Age'].median()
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, AgeByPclassAndSex[sex][pclass]))
print('Median age of all passengers: {}'.format(dfAll['Age'].median()))
dfAll['Age'] = dfAll.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

print(dfAll[dfAll['Embarked'].isnull()])
dfAll['Embarked'] = dfAll['Embarked'].fillna('S')

print(dfAll[dfAll['Fare'].isnull()])
medFare = dfAll.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
dfAll['Fare'] = dfAll['Fare'].fillna(medFare)

dfAll['Deck'] = dfAll['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

dfAllDecks = dfAll.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()

# def getPclassDistance(df):
#     deckCounts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
#     decks = df.columns.levels[0]    
    
#     for deck in decks:
#         for pclass in range(1, 4):
#             try:
#                 count = df[deck][pclass].iloc[0]
#                 deckCounts[deck][pclass] = count 
#             except KeyError:
#                 deckCounts[deck][pclass] = 0
                
#     dfDecks = pd.DataFrame(deckCounts)    
#     deckPercentages = {}

#     for col in dfDecks.columns:
#         deckPercentages[col] = [(count / dfDecks[col].sum()) * 100 for count in dfDecks[col]]
        
#     return deckCounts, deckPercentages

# def displayPclassDistance(percentages):
    
#     dfPercentages = pd.DataFrame(percentages).transpose()
#     deckNames = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
#     barCount = np.arange(len(deckNames))  
#     barWidth = 0.85
    
#     pclass1 = dfPercentages[0]
#     pclass2 = dfPercentages[1]
#     pclass3 = dfPercentages[2]
    
#     plt.figure(figsize=(15, 10))
#     plt.bar(barCount, pclass1, color='#b5ffb9', edgecolor='white', width=barWidth, label='Passenger Class 1')
#     plt.bar(barCount, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=barWidth, label='Passenger Class 2')
#     plt.bar(barCount, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=barWidth, label='Passenger Class 3')

#     plt.xlabel('Deck', size=15, labelpad=20)
#     plt.ylabel('Passenger Class Percentage', size=10, labelpad=20)
#     plt.xticks(barCount, deckNames)    
#     plt.tick_params(axis='x', labelsize=15)
#     plt.tick_params(axis='y', labelsize=15)
    
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 7})
#     plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   
    
#     plt.show()    

# allDecksCount, allDeckPer = getPclassDistance(dfAllDecks)
# displayPclassDistance(allDeckPer)

# 1 passenger in T deck, wont hurt to change to A
idx = dfAll[dfAll['Deck'] == 'T'].index
dfAll.loc[idx, 'Deck'] = 'A'

dfAllSurvivingDecks = dfAll.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

# def getSurvivedDistance(df):
#     survivalCount = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
#     decks = df.columns.levels[0]    

#     for deck in decks:
#         for survive in range(0, 2):
#             survivalCount[deck][survive] = df[deck][survive].iloc[0]
            
#     dfSurvived = pd.DataFrame(survivalCount)
#     survivalPercent = {}

#     for col in dfSurvived.columns:
#         survivalPercent[col] = [(count / dfSurvived[col].sum()) * 100 for count in dfSurvived[col]]
        
#     return survivalCount, survivalPercent

# def displaySurvivalDistance(percentages):
#     dfSurvivedPercentages = pd.DataFrame(percentages).transpose()
#     deckNames = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
#     barCount = np.arange(len(deckNames))  
#     barWidth = 0.85    

#     Dead = dfSurvivedPercentages[0]
#     Alive = dfSurvivedPercentages[1]
    
#     plt.figure(figsize=(15, 10))
#     plt.bar(barCount, Dead, color='#b5ffb9', edgecolor='white', width=barWidth, label="Not Survived")
#     plt.bar(barCount, Alive, bottom=Dead, color='#f9bc86', edgecolor='white', width=barWidth, label="Survived")
 
#     plt.xlabel('Deck', size=15, labelpad=20)
#     plt.ylabel('Survival Percentage', size=15, labelpad=20)
#     plt.xticks(barCount, deckNames)    
#     plt.tick_params(axis='x', labelsize=15)
#     plt.tick_params(axis='y', labelsize=15)
    
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 7})
#     plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
#     plt.show()

# allSurvivedCount, allSurvivedPer = getSurvivedDistance(dfAllSurvivingDecks)
# displaySurvivalDistance(allSurvivedPer)

dfAll['Deck'] = dfAll['Deck'].replace(['A', 'B', 'C'], 'ABC')
dfAll['Deck'] = dfAll['Deck'].replace(['D', 'E'], 'DE')
dfAll['Deck'] = dfAll['Deck'].replace(['F', 'G'], 'FG')
print(dfAll['Deck'].value_counts())
dfAll.drop(['Cabin'], inplace=True, axis=1)

dfTrainFinal, dfTestFinal = divideDf(dfAll)
dfTrainFinal.to_csv('cleanedtrain.csv', index=False) 
dfTestFinal.to_csv('cleanedtest.csv', index=False) 

dfs = [dfTrainFinal, dfTestFinal]
for df in dfs:
    displayMissing(df)

























































































































# def cleantrainingset(path= 'train.csv'):
#     trainingdf = pd.read_csv('train.csv')
#     medianage = trainingdf["Age"].median()
#     trainingdf["Age"].fillna(medianage, inplace=True)
#     trainingdf["Cabin"].fillna("U", inplace=True)
#     trainingdf["CabinOnDeck"] = trainingdf["Cabin"].apply(lambda i : i[0] if isinstance(i, str) and i != "U" else "U")

#     if trainingdf["Embarked"].isnull().any():
#         kNN = 5;  
#         commonport = trainingdf["Embarked"].mode().iloc[0]  ##iloc[0] mai pehli value is chosen, agar mode mai tie, it chooses pehli value from mode
#         for i in trainingdf[trainingdf["Embarked"].isnull()].index:
#             Pclass = trainingdf.at[i, "Pclass"]
#             fare = trainingdf.at[i, "Fare"]
#             potentialneighbours = trainingdf[(trainingdf["Pclass"] == Pclass) & (trainingdf["Fare"].notnull()) & (trainingdf["Embarked"].notnull())].copy()
#             if potentialneighbours.empty:
#                 choice = commonport
#             else:
#                 potentialneighbours["fareDifference"] = (potentialneighbours["Fare"] - fare).abs()
#                 nearestneighbour = potentialneighbours.nsmallest(kNN, "fareDifference")
#                 if not nearestneighbour.empty:
#                     choice = nearestneighbour["Embarked"].mode().iloc[0]
#                 else:
#                     choice = commonport
#                 trainingdf.at[i, "Embarked"] = choice             
#     trainingdf["Embarked"].fillna(trainingdf["Embarked"].mode().iloc[0], inplace=True)  #if still missing values, replace them w most common port ki values

#     trainingdf["Sex"] = trainingdf["Sex"].map({"male" : 0, "female" : 1}).astype(int)
#     trainingdf["Embarked"] = trainingdf["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2}).astype(int)
#     trainingdf["CabinOnDeck"] = pd.factorize(trainingdf["CabinOnDeck"])[0]

#     cleanedfeatures = trainingdf[["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "CabinOnDeck", "Embarked"]].copy()
#     cleanedfeatures.to_csv("cleanedtraincsv", index=False)
#     return cleanedfeatures

# def cleaningtestset(path= 'test.csv'):
#     testdf = pd.read_csv('test.csv')
#     testdf["Age"] = testdf["Age"].fillna(testdf["Age"].median())

#     if 'Cabin' not in testdf.columns:   #incase cabins column wapis urrjae
#         testdf["Cabin"] = "U"
#     else:
#         testdf["Cabin"].fillna("U")
#     testdf["CabinOnDeck"] = testdf["Cabin"].apply(lambda i : i[0] if isinstance(i, str) and i != "U" else "U")

#     if testdf["Embarked"].isnull().any():
#         testdf["Embarked"] = testdf["Embarked"].fillna(testdf["Embarked"].mode().iloc[0])
#     testdf["Fare"] = testdf["Fare"].fillna(testdf["Fare"].median())

#     testdf["Sex"] = testdf["Sex"].map({"male" : 0, "female" : 1}).astype(int)
#     testdf["Embarked"] = testdf["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2}).astype(int)
#     testdf["CabinOnDeck"] = pd.factorize(testdf["CabinOnDeck"])[0]
#     cleanedtestfeatures = testdf[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "CabinOnDeck", "Embarked"]].copy()
#     cleanedtestfeatures.to_csv("cleanedtestcsv", index=False)
#     return cleanedtestfeatures