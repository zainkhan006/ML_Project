import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

def concatenateDf(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

dfTrain = pd.read_csv('train.csv')
print(dfTrain.head(20))
dfTest = pd.read_csv('test.csv')
print("\n", dfTest.head(20))

dfTrain.name = 'Training Set'
dfTest.name = 'Test Set'
dfAll = concatenateDf(dfTrain, dfTest)
dfAll['Deck'] = dfAll['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
dfAllDecks = dfAll.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()

def getPclassDistance(df):
    deckCounts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]    
    
    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass].iloc[0]
                deckCounts[deck][pclass] = count 
            except KeyError:
                deckCounts[deck][pclass] = 0
                
    dfDecks = pd.DataFrame(deckCounts)    
    deckPercentages = {}

    for col in dfDecks.columns:
        deckPercentages[col] = [(count / dfDecks[col].sum()) * 100 for count in dfDecks[col]]
        
    return deckCounts, deckPercentages

def displayPclassDistance(percentages):
    
    dfPercentages = pd.DataFrame(percentages).transpose()
    deckNames = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    barCount = np.arange(len(deckNames))  
    barWidth = 0.85
    
    pclass1 = dfPercentages[0]
    pclass2 = dfPercentages[1]
    pclass3 = dfPercentages[2]
    
    plt.figure(figsize=(15, 10))
    plt.bar(barCount, pclass1, color='#b5ffb9', edgecolor='white', width=barWidth, label='Passenger Class 1')
    plt.bar(barCount, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=barWidth, label='Passenger Class 2')
    plt.bar(barCount, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=barWidth, label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=10, labelpad=20)
    plt.xticks(barCount, deckNames)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 7})
    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   
    
    plt.show()    

allDecksCount, allDeckPer = getPclassDistance(dfAllDecks)
displayPclassDistance(allDeckPer)

idx = dfAll[dfAll['Deck'] == 'T'].index
dfAll.loc[idx, 'Deck'] = 'A'

dfAllSurvivingDecks = dfAll.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

def getSurvivedDistance(df):
    survivalCount = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
    decks = df.columns.levels[0]    

    for deck in decks:
        for survive in range(0, 2):
            survivalCount[deck][survive] = df[deck][survive].iloc[0]
            
    dfSurvived = pd.DataFrame(survivalCount)
    survivalPercent = {}

    for col in dfSurvived.columns:
        survivalPercent[col] = [(count / dfSurvived[col].sum()) * 100 for count in dfSurvived[col]]
        
    return survivalCount, survivalPercent

def displaySurvivalDistance(percentages):
    dfSurvivedPercentages = pd.DataFrame(percentages).transpose()
    deckNames = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    barCount = np.arange(len(deckNames))  
    barWidth = 0.85    

    Dead = dfSurvivedPercentages[0]
    Alive = dfSurvivedPercentages[1]
    
    plt.figure(figsize=(15, 10))
    plt.bar(barCount, Dead, color='#b5ffb9', edgecolor='white', width=barWidth, label="Not Survived")
    plt.bar(barCount, Alive, bottom=Dead, color='#f9bc86', edgecolor='white', width=barWidth, label="Survived")
 
    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(barCount, deckNames)    
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 7})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)
    
    plt.show()

allSurvivedCount, allSurvivedPer = getSurvivedDistance(dfAllSurvivingDecks)
displaySurvivalDistance(allSurvivedPer)