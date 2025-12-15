hyperparameters = {
    'Decision Tree': {
        'best_accuracy': 0.8268,  # Optimized with GridSearchCV
        'best_params': {
            'criterion': 'gini',
            'max_depth': 6,
            'max_features': None,
            'min_samples_leaf': 2,
            'min_samples_split': 10
        }
    },
    'Random Forest': {
        'best_accuracy': 0.8045,
        'best_params': {
            'criterion': 'gini',
            'max_depth': 5,
            'max_features': 'log2',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 300
        }
    },
    'Logistic Regression': {
        'best_accuracy': 0.8212,
        'best_params': {
            'C': 0.1,
            'max_iter': 1000,
            'penalty': 'l2',
            'solver': 'liblinear'
        }
    },
    'SVM': {
        'best_accuracy': 0.8212,
        'best_params': {
            'C': 1,
            'gamma': 'auto',
            'kernel': 'rbf'
        }
    },
    'KNN': {
        'best_accuracy': 0.8156,
        'best_params': {
            'metric': 'manhattan',
            'n_neighbors': 9,
            'weights': 'uniform'
        }
    }
}