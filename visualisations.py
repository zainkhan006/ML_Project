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
dfTrain['Deck'] = dfTrain['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
dfTest['Deck'] = dfTest['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
dfAllDecks = dfAll.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 
                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()

####################################################### preprocessing visualisations #########################################################################


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

survived = dfTrain['Survived'].value_counts()[1]
notSurvived = dfTrain['Survived'].value_counts()[0]
survivedPersons = survived / dfTrain.shape[0] * 100
DeadPersons = notSurvived / dfTrain.shape[0] * 100

print('{} of {} passengers survived and it is {:.2f}% of the training set.'.format(survived, dfTrain.shape[0], survivedPersons))
print('{} of {} passengers didnt survive and it is {:.2f}% of the training set.'.format(notSurvived, dfTrain.shape[0], DeadPersons))

plt.figure(figsize=(10, 8))
sns.countplot(data=dfTrain, x='Survived')

plt.xlabel('Survival', size=15, labelpad=15)
plt.ylabel('Passenger Count', size=15, labelpad=15)
plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(DeadPersons), 'Survived ({0:.2f}%)'.format(survivedPersons)])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)
plt.title('Training Set Survival Distribution', size=15, y=1.05)
plt.show()

df_train_corr = dfTrain.drop(['PassengerId'], axis=1).corr(numeric_only=True).abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

df_test_corr = dfTest.corr(numeric_only=True).abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)

traincorr = df_train_corr_nd['Correlation Coefficient'] > 0.1
print(df_train_corr_nd[traincorr])

testcorr = df_test_corr_nd['Correlation Coefficient'] > 0.1
print(df_test_corr_nd[testcorr])

fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

sns.heatmap(dfTrain.drop(['PassengerId'], axis=1).corr(numeric_only=True), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 10})
sns.heatmap(dfTest.drop(['PassengerId'], axis=1).corr(numeric_only=True), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 10})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=10)
    axs[i].tick_params(axis='y', labelsize=10)
    
axs[0].set_title('Training Set Correlations', size=10)
axs[1].set_title('Test Set Correlations', size=10)

plt.show()

cont_features = ['Age', 'Fare']
surv = dfTrain['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):    
    # Distribution of survival in feature
    sns.histplot(dfTrain[~surv][feature], label='Not Survived', color='#e74c3c', ax=axs[0][i], kde=True, stat='density')
    sns.histplot(dfTrain[surv][feature], label='Survived', color='#2ecc71', ax=axs[0][i], kde=True, stat='density')
    
    # Distribution of feature in dataset
    sns.histplot(dfTrain[feature], label='Training Set', color='#e74c3c', ax=axs[1][i])
    sns.histplot(dfTest[feature], label='Test Set', color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=10)
        axs[i][j].tick_params(axis='y', labelsize=10)
    
    axs[0][i].legend(loc='upper right', prop={'size': 10})
    axs[1][i].legend(loc='upper right', prop={'size': 10})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=10, y=1.00)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=10, y=1.0)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=10, y=1.0)
plt.tight_layout()      
plt.show()

cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(10, 10))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=dfTrain)
    
    plt.xlabel('{}'.format(feature), size=10, labelpad=15)
    plt.ylabel('Passenger Count', size=10, labelpad=15)    
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 10})
    plt.title('Count of Survival in {} Feature'.format(feature), size=10, y=1.00)

plt.show()

#################################################feature engineering visualisations###################################################################