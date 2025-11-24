import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessingData():
    df = pd.read_csv('cleanedtrain.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
    df['Deck'] = df['Deck'].map({'M': 0, 'ABC': 1, 'DE': 2, 'FG': 3}).astype(int)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Deck', 'Embarked']
    target = 'Survived'
    
    X = df[features].copy()
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    numericalFeatures = ['Age', 'Fare']
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[numericalFeatures] = scaler.fit_transform(X_train[numericalFeatures])
    X_test[numericalFeatures] = scaler.transform(X_test[numericalFeatures])
    
    return X_train, X_test, y_train, y_test, scaler, features