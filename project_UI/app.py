import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocessingData
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#################################################### initialisation #############################################################################################

st.set_page_config(page_title="ML Learning Tool", layout="wide")
st.title("Machine Learning interactive website")

@st.cache_data
def get_data():
    return preprocessingData()

X_train, X_test, y_train, y_test, scaler, features = get_data()
c1, c2, c3 = st.columns(3)
c1.metric("Total Passengers", len(X_train) + len(X_test))
c2.metric("Training Samples", len(X_train))
c3.metric("Test Samples", len(X_test))
st.divider()

st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Decision Tree", "Logistic Regression", "Random Forest", "SVM", "KNN"]
)

######################################################### including dt #########################################################################################
if model_choice == "Decision Tree":
    st.sidebar.markdown("### Hyperparameters")
    
    # Hyperparameter controls
    maxDepth = st.sidebar.slider(
        "Max Depth",
        min_value=1,
        max_value=20, #can increase, 20 is a decent start
        value=5,  #default is none, but 5 is reasonable
        help="Maximum depth of tree. Deeper trees can lead to more complex patterns being spotted. Deeper trees may lead to overfitting"
    )
    
    minSamplesSplit = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,  # sklearn default
        help="Minimum number of samples required to split an internal node."
    )
    
    criterion = st.sidebar.selectbox(
        "Criterion",
        ["gini", "entropy"],
        help="Function which measures split quality. Gini for Gini impurity, Entropy for information gain."
    )
    
    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training Decision Tree"):
            dtModel = DecisionTreeClassifier(
                max_depth=maxDepth,
                min_samples_split=minSamplesSplit,
                criterion=criterion,
                random_state=2025
            )
            dtModel.fit(X_train, y_train)
            yTrainPred = dtModel.predict(X_train)
            yTestPred = dtModel.predict(X_test)
            
            trainingAccuracy = accuracy_score(y_train, yTrainPred)
            testingAccuracy = accuracy_score(y_test, yTestPred)
            accuracyGap = trainingAccuracy - testingAccuracy
            
            st.success("Model trained successfully!")
            st.markdown("### Model Performance:")
            c1, c2, c3 = st.columns(3)
            
            c1.metric(
                "Training Accuracy",
                f"{trainingAccuracy:.2%}",
                help="Accuracy on training data"
            )
            c2.metric(
                "Test Accuracy",
                f"{testingAccuracy:.2%}",
                help="Accuracy on unseen test data"
            )
            c3.metric(
                "Accuracy Gap",
                f"{accuracyGap:.2%}",
                delta=f"{-accuracyGap:.2%}",
                delta_color="inverse",
                help="Difference between training and test accuracy. Lower accuracy gap is better"
            )
            
            st.markdown("### Interpretation:")
            if accuracyGap > 0.15:
                st.error("**High levels of overfitting detected!!!** "
                        "Your model memorizes training data but doesn't perform well on testing data, meaning it does not generalize well. "
                        "To reduce overfitting, try reducing max_depth or increasing min_samples_split.")
            elif accuracyGap > 0.10:
                st.warning("**Moderate Overfitting.** " \
                            "The model performs significantly better on training data than it does on test data. " \
                            "Consider making the tree simpler by reducing max_depth")
            elif testingAccuracy < 0.70:
                st.warning("**Low Accuracy.** " \
                            "Your model is underfitting. " \
                            "Try increasing max_depth or using a different criterion.")
            elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                st.success("**Good job!** " \
                            "Your model generalizes well with good accuracy.")
            else:
                st.info("**Great job!** " \
                        "Keep experimenting with hyperparameters to improve further.")
            
            st.divider()
            
            # Confusion matrix
            st.markdown("### Confusion Matrix (Test Set)")

            # Explanation of confusion matrix
            with st.expander("Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            st.divider()
            
            # Feature Importance
            st.markdown("### 📈 Feature Importance")

            #explaining feature importance
            with st.expander("ℹ️ Understanding Feature Importance"):
                st.markdown("""
                Feature importance shows which features the decision tree used most for making decisions.
                Higher values mean the feature was more influential in the model's predictions.
                
                For the Titanic dataset:
                - **Sex** is usually most important (women had higher survival rates)
                - **Pclass** (passenger class) also matters significantly
                - **Age** and **Fare** can provide additional discrimination
                """)
            
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': dtModel.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=feature_imp,
                x='Importance',
                y='Feature',
                palette='viridis',
                ax=ax
            )
            ax.set_title('Feature Importance in Decision Tree', fontsize=14)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            st.pyplot(fig)
            plt.close()
                        
            st.divider()
            
            # Detailed metrics
            st.markdown("### 📋 Detailed Classification Report")
            report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

#########################################################random forest######################################################################################
elif model_choice == "Random Forest":
    st.sidebar.markdown("### Hyperparameters")

        # Hyperparameter controls
    maxDepth = st.sidebar.slider(
        "Max Depth",
        min_value=1,
        max_value=20, #can increase, 20 is a decent start
        value=5,  #default is none, but 5 is reasonable
        help="Maximum depth of tree. Deeper trees can lead to more complex patterns being spotted. Deeper trees may lead to overfitting"
    )

    minSamplesSplit = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,  # sklearn default
        help="Minimum number of samples required to split an internal node."
    )
    
    criterion = st.sidebar.selectbox(
        "Criterion",
        ["gini", "entropy"],
        help="Function which measures split quality. Gini for Gini impurity, Entropy for information gain."
    )

    n_estimators = st.sidebar.slider(
        "Number of decision trees ",
        min_value=50,
        max_value=500,
        value=100, 
        step=50,
        help="Number of decision trees in the forest. More trees generally improve performance but increase computation time."
    )

    minSamplesLeaf = st.sidebar.slider(
        "Minimum Samples to be a leaf",
        min_value=1,
        max_value=10,
        value=1,
        help="Minimum number of samples required to be at a leaf node."
    )

    maxFeatures = st.sidebar.selectbox(
        "Maximum features",
        ["sqrt", "log2", "None"],
        index=0,
        help="Number of features to consider when looking for the best split. 'sqrt' and 'log2' help reduce overfitting."
    )

    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training Random Forest"):
            maxFeaturesValue = None if maxFeatures == "None" else maxFeatures

            rfModel = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=maxDepth,
                min_samples_split=minSamplesSplit,
                min_samples_leaf=minSamplesLeaf,
                max_features=maxFeaturesValue,
                criterion=criterion,
                random_state=2025
            )
            rfModel.fit(X_train, y_train)
            yTrainPred = rfModel.predict(X_train)
            yTestPred = rfModel.predict(X_test)
            
            trainingAccuracy = accuracy_score(y_train, yTrainPred)
            testingAccuracy = accuracy_score(y_test, yTestPred)
            accuracyGap = trainingAccuracy - testingAccuracy
            
            st.success("Model trained successfully!")
            st.markdown("### Model Performance")
            c1, c2, c3 = st.columns(3)
            
            c1.metric(
                "Training Accuracy",
                f"{trainingAccuracy:.2%}",
                help="Accuracy on training data"
            )
            c2.metric(
                "Test Accuracy",
                f"{testingAccuracy:.2%}",
                help="Accuracy on unseen test data"
            )
            c3.metric(
                "Accuracy Gap",
                f"{accuracyGap:.2%}",
                delta=f"{-accuracyGap:.2%}",
                delta_color="inverse",
                help="Difference between training and test accuracy. Lower accuracy gap is better"
            )
            
            st.markdown("### Interpretation:")
            if accuracyGap > 0.15:
                st.error("**High levels of overfitting detected!!!** "
                        "Your model memorizes training data but doesn't perform well on testing data, meaning it does not generalize well. "
                        "To reduce overfitting, try reducing max_depth or increasing min_samples_split.")
            elif accuracyGap > 0.10:
                st.warning("**Moderate Overfitting.** " \
                            "The model performs significantly better on training data than it does on test data. " \
                            "Consider making the tree simpler by reducing max_depth")
            elif testingAccuracy < 0.70:
                st.warning("**Low Accuracy.** " \
                            "Your model is underfitting. " \
                            "Try increasing max_depth or using a different criterion.")
            elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                st.success("**Good job!** " \
                            "Your model generalizes well with good accuracy.")
            else:
                st.info("**Great job!** " \
                        "Keep experimenting with hyperparameters to improve further.")
            
            st.divider()
            
            # Confusion Matrix
            st.markdown("### 🔢 Confusion Matrix (Test Set)")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Greens',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            
            # Explanation of confusion matrix
            with st.expander("ℹ️ Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            
            st.divider()
            
            # Confusion matrix
            st.markdown("### Confusion Matrix (Test Set)")

            # Explanation of confusion matrix
            with st.expander("Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            st.divider()
            
            # Feature Importance
            st.markdown("### 📈 Feature Importance")

            #explaining feature importance
            with st.expander("Understanding Feature Importance"):
                st.markdown("""
                Feature importance shows which features the decision tree used most for making decisions.
                Higher values mean the feature was more influential in the model's predictions.
                
                For the Titanic dataset:
                - **Sex** is usually most important (women had higher survival rates)
                - **Pclass** (passenger class) also matters significantly
                - **Age** and **Fare** can provide additional discrimination
                """)
            
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': rfModel.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=feature_imp,
                x='Importance',
                y='Feature',
                palette='viridis',
                ax=ax
            )
            ax.set_title('Feature Importance in Decision Tree', fontsize=14)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            st.pyplot(fig)
            plt.close()
                        
            st.divider()
            
            # Detailed metrics
            st.markdown("### 📋 Detailed Classification Report")
            report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

#################################################logistic regression########################################################################################
elif model_choice == "Logistic Regression":
    st.sidebar.markdown("### Logistic Regression Hyperparameters")
    
    # Hyperparameter controls
    C = st.sidebar.slider(
        "Regularization Strength",
        min_value=0.001,
        max_value=100.0,
        value=1.0,
        step=0.001,
        format="%.3f",
        help="Inverse of regularization strength. Smaller values mean stronger regularization (simpler model)."
    )
    
    penalty = st.sidebar.selectbox(
        "Penalty",
        ["l2", "l1"],
        help="Type of regularization. L2 (Ridge) is generally preferred; L1 (Lasso) can zero out features."
    )
    
    solver = st.sidebar.selectbox(
        "Solver",
        ["liblinear", "lbfgs", "saga"],
        help="Algorithm to use for optimization. 'liblinear' works well for small datasets."
    )
    
    max_iter = st.sidebar.slider(
        "Max Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Maximum number of iterations for the solver to converge."
    )
    
    # Train button
    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training Logistic Regression..."):
            lrModel = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                random_state=42
            )
            
            try:
                lrModel.fit(X_train, y_train)
    
                yTrainPred = lrModel.predict(X_train)
                yTestPred = lrModel.predict(X_test)
                
                trainingAccuracy = accuracy_score(y_train, yTrainPred)
                testingAccuracy = accuracy_score(y_test, yTestPred)
                accuracyGap = trainingAccuracy - testingAccuracy
            
                st.success("Model trained successfully!")
                st.markdown("### Model Performance")
                c1, c2, c3 = st.columns(3)
                
                c1.metric(
                    "Training Accuracy",
                    f"{trainingAccuracy:.2%}",
                    help="Accuracy on training data"
                )
                c2.metric(
                    "Test Accuracy",
                    f"{testingAccuracy:.2%}",
                    help="Accuracy on unseen test data"
                )
                c3.metric(
                    "Accuracy Gap",
                    f"{accuracyGap:.2%}",
                    delta=f"{-accuracyGap:.2%}",
                    delta_color="inverse",
                    help="Difference between training and test accuracy. Lower is better!"
                )
                
                st.markdown("### Interpretation:")
                if accuracyGap > 0.15:
                    st.error("**High levels of overfitting detected!!!** "
                            "Your model memorizes training data but doesn't perform well on testing data, meaning it does not generalize well. "
                            "To reduce overfitting, try reducing max_depth or increasing min_samples_split.")
                elif accuracyGap > 0.10:
                    st.warning("**Moderate Overfitting.** " \
                                "The model performs significantly better on training data than it does on test data. " \
                                "Consider making the tree simpler by reducing max_depth")
                elif testingAccuracy < 0.70:
                    st.warning("**Low Accuracy.** " \
                                "Your model is underfitting. " \
                                "Try increasing max_depth or using a different criterion.")
                elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                    st.success("**Good job!** " \
                                "Your model generalizes well with good accuracy.")
                else:
                    st.info("**Great job!** " \
                            "Keep experimenting with hyperparameters to improve further.")
                
                st.divider()
                
                # Confusion Matrix
                st.markdown("### 🔢 Confusion Matrix (Test Set)")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Purples',
                    xticklabels=['Died', 'Survived'],
                    yticklabels=['Died', 'Survived'],
                    ax=ax,
                    cbar_kws={'label': 'Count'}
                )
                ax.set_xlabel('Predicted', fontsize=12)
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
                
                st.pyplot(fig)
                plt.close()
                
                with st.expander("ℹ️ Understanding the Confusion Matrix"):
                    st.markdown("""
                    - **Top-left (True Negative):** Correctly predicted deaths
                    - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                    - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                    - **Bottom-right (True Positive):** Correctly predicted survival
                    
                    A good model has high numbers on the diagonal (top-left and bottom-right).
                    """)
                
                st.divider()
                
                # Detailed metrics
                st.markdown("### Detailed Classification Report")
                report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.info("💡 Try changing the penalty/solver combination. Some combinations are incompatible (e.g., L1 penalty requires 'liblinear' or 'saga' solver).")

##############################################################SVM##############################################################################################
elif model_choice == "Random Forest":
    st.sidebar.markdown("### Hyperparameters")

        # Hyperparameter controls
    maxDepth = st.sidebar.slider(
        "Max Depth",
        min_value=1,
        max_value=20, #can increase, 20 is a decent start
        value=5,  #default is none, but 5 is reasonable
        help="Maximum depth of tree. Deeper trees can lead to more complex patterns being spotted. Deeper trees may lead to overfitting"
    )

    minSamplesSplit = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,  # sklearn default
        help="Minimum number of samples required to split an internal node."
    )
    
    criterion = st.sidebar.selectbox(
        "Criterion",
        ["gini", "entropy"],
        help="Function which measures split quality. Gini for Gini impurity, Entropy for information gain."
    )

    n_estimators = st.sidebar.slider(
        "Number of decision trees ",
        min_value=50,
        max_value=500,
        value=100, 
        step=50,
        help="Number of decision trees in the forest. More trees generally improve performance but increase computation time."
    )

    minSamplesLeaf = st.sidebar.slider(
        "Minimum Samples to be a leaf",
        min_value=1,
        max_value=10,
        value=1,
        help="Minimum number of samples required to be at a leaf node."
    )

    maxFeatures = st.sidebar.selectbox(
        "Maximum features",
        ["sqrt", "log2", "None"],
        index=0,
        help="Number of features to consider when looking for the best split. 'sqrt' and 'log2' help reduce overfitting."
    )

    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training Random Forest"):
            maxFeaturesValue = None if maxFeatures == "None" else maxFeatures

            rfModel = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=maxDepth,
                min_samples_split=minSamplesSplit,
                min_samples_leaf=minSamplesLeaf,
                max_features=maxFeaturesValue,
                criterion=criterion,
                random_state=2025
            )
            rfModel.fit(X_train, y_train)
            yTrainPred = rfModel.predict(X_train)
            yTestPred = rfModel.predict(X_test)
            
            trainingAccuracy = accuracy_score(y_train, yTrainPred)
            testingAccuracy = accuracy_score(y_test, yTestPred)
            accuracyGap = trainingAccuracy - testingAccuracy
            
            st.success("Model trained successfully!")
            st.markdown("### Model Performance")
            c1, c2, c3 = st.columns(3)
            
            c1.metric(
                "Training Accuracy",
                f"{trainingAccuracy:.2%}",
                help="Accuracy on training data"
            )
            c2.metric(
                "Test Accuracy",
                f"{testingAccuracy:.2%}",
                help="Accuracy on unseen test data"
            )
            c3.metric(
                "Accuracy Gap",
                f"{accuracyGap:.2%}",
                delta=f"{-accuracyGap:.2%}",
                delta_color="inverse",
                help="Difference between training and test accuracy. Lower accuracy gap is better"
            )
            
            st.markdown("### Interpretation:")
            if accuracyGap > 0.15:
                st.error("**High levels of overfitting detected!!!** "
                        "Your model memorizes training data but doesn't perform well on testing data, meaning it does not generalize well. "
                        "To reduce overfitting, try reducing max_depth or increasing min_samples_split.")
            elif accuracyGap > 0.10:
                st.warning("**Moderate Overfitting.** " \
                            "The model performs significantly better on training data than it does on test data. " \
                            "Consider making the tree simpler by reducing max_depth")
            elif testingAccuracy < 0.70:
                st.warning("**Low Accuracy.** " \
                            "Your model is underfitting. " \
                            "Try increasing max_depth or using a different criterion.")
            elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                st.success("**Good job!** " \
                            "Your model generalizes well with good accuracy.")
            else:
                st.info("**Great job!** " \
                        "Keep experimenting with hyperparameters to improve further.")
            
            st.divider()
            
            # Confusion Matrix
            st.markdown("### 🔢 Confusion Matrix (Test Set)")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Greens',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            
            # Explanation of confusion matrix
            with st.expander("ℹ️ Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            
            st.divider()
            
            # Confusion matrix
            st.markdown("### Confusion Matrix (Test Set)")

            # Explanation of confusion matrix
            with st.expander("Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            st.divider()
            
            # Feature Importance
            st.markdown("### 📈 Feature Importance")

            #explaining feature importance
            with st.expander("Understanding Feature Importance"):
                st.markdown("""
                Feature importance shows which features the decision tree used most for making decisions.
                Higher values mean the feature was more influential in the model's predictions.
                
                For the Titanic dataset:
                - **Sex** is usually most important (women had higher survival rates)
                - **Pclass** (passenger class) also matters significantly
                - **Age** and **Fare** can provide additional discrimination
                """)
            
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': rfModel.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=feature_imp,
                x='Importance',
                y='Feature',
                palette='viridis',
                ax=ax
            )
            ax.set_title('Feature Importance in Decision Tree', fontsize=14)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            st.pyplot(fig)
            plt.close()
                        
            st.divider()
            
            # Detailed metrics
            st.markdown("### 📋 Detailed Classification Report")
            report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)


######################################################### KNN ########################################################################################
    
elif model_choice == "Random Forest":
    st.sidebar.markdown("### Hyperparameters")

        # Hyperparameter controls
    maxDepth = st.sidebar.slider(
        "Max Depth",
        min_value=1,
        max_value=20, #can increase, 20 is a decent start
        value=5,  #default is none, but 5 is reasonable
        help="Maximum depth of tree. Deeper trees can lead to more complex patterns being spotted. Deeper trees may lead to overfitting"
    )

    minSamplesSplit = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,  # sklearn default
        help="Minimum number of samples required to split an internal node."
    )
    
    criterion = st.sidebar.selectbox(
        "Criterion",
        ["gini", "entropy"],
        help="Function which measures split quality. Gini for Gini impurity, Entropy for information gain."
    )

    n_estimators = st.sidebar.slider(
        "Number of decision trees ",
        min_value=50,
        max_value=500,
        value=100, 
        step=50,
        help="Number of decision trees in the forest. More trees generally improve performance but increase computation time."
    )

    minSamplesLeaf = st.sidebar.slider(
        "Minimum Samples to be a leaf",
        min_value=1,
        max_value=10,
        value=1,
        help="Minimum number of samples required to be at a leaf node."
    )

    maxFeatures = st.sidebar.selectbox(
        "Maximum features",
        ["sqrt", "log2", "None"],
        index=0,
        help="Number of features to consider when looking for the best split. 'sqrt' and 'log2' help reduce overfitting."
    )

    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training Random Forest"):
            maxFeaturesValue = None if maxFeatures == "None" else maxFeatures

            rfModel = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=maxDepth,
                min_samples_split=minSamplesSplit,
                min_samples_leaf=minSamplesLeaf,
                max_features=maxFeaturesValue,
                criterion=criterion,
                random_state=2025
            )
            rfModel.fit(X_train, y_train)
            yTrainPred = rfModel.predict(X_train)
            yTestPred = rfModel.predict(X_test)
            
            trainingAccuracy = accuracy_score(y_train, yTrainPred)
            testingAccuracy = accuracy_score(y_test, yTestPred)
            accuracyGap = trainingAccuracy - testingAccuracy
            
            st.success("Model trained successfully!")
            st.markdown("### Model Performance")
            c1, c2, c3 = st.columns(3)
            
            c1.metric(
                "Training Accuracy",
                f"{trainingAccuracy:.2%}",
                help="Accuracy on training data"
            )
            c2.metric(
                "Test Accuracy",
                f"{testingAccuracy:.2%}",
                help="Accuracy on unseen test data"
            )
            c3.metric(
                "Accuracy Gap",
                f"{accuracyGap:.2%}",
                delta=f"{-accuracyGap:.2%}",
                delta_color="inverse",
                help="Difference between training and test accuracy. Lower accuracy gap is better"
            )
            
            st.markdown("### Interpretation:")
            if accuracyGap > 0.15:
                st.error("**High levels of overfitting detected!!!** "
                        "Your model memorizes training data but doesn't perform well on testing data, meaning it does not generalize well. "
                        "To reduce overfitting, try reducing max_depth or increasing min_samples_split.")
            elif accuracyGap > 0.10:
                st.warning("**Moderate Overfitting.** " \
                            "The model performs significantly better on training data than it does on test data. " \
                            "Consider making the tree simpler by reducing max_depth")
            elif testingAccuracy < 0.70:
                st.warning("**Low Accuracy.** " \
                            "Your model is underfitting. " \
                            "Try increasing max_depth or using a different criterion.")
            elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                st.success("**Good job!** " \
                            "Your model generalizes well with good accuracy.")
            else:
                st.info("**Great job!** " \
                        "Keep experimenting with hyperparameters to improve further.")
            
            st.divider()
            
            # Confusion Matrix
            st.markdown("### 🔢 Confusion Matrix (Test Set)")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Greens',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            
            # Explanation of confusion matrix
            with st.expander("ℹ️ Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            
            st.divider()
            
            # Confusion matrix
            st.markdown("### Confusion Matrix (Test Set)")

            # Explanation of confusion matrix
            with st.expander("Understanding the Confusion Matrix"):
                st.markdown("""
                - **Top-left (True Negative):** Correctly predicted deaths
                - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                - **Bottom-right (True Positive):** Correctly predicted survival
                
                A good model has high numbers on the diagonal (top-left and bottom-right).
                """)
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, yTestPred)
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title(f'Confusion Matrix - Test Accuracy: {testingAccuracy:.2%}', fontsize=14)
            
            st.pyplot(fig)
            plt.close()
            st.divider()
            
            # Feature Importance
            st.markdown("### 📈 Feature Importance")

            #explaining feature importance
            with st.expander("Understanding Feature Importance"):
                st.markdown("""
                Feature importance shows which features the decision tree used most for making decisions.
                Higher values mean the feature was more influential in the model's predictions.
                
                For the Titanic dataset:
                - **Sex** is usually most important (women had higher survival rates)
                - **Pclass** (passenger class) also matters significantly
                - **Age** and **Fare** can provide additional discrimination
                """)
            
            feature_imp = pd.DataFrame({
                'Feature': features,
                'Importance': rfModel.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=feature_imp,
                x='Importance',
                y='Feature',
                palette='viridis',
                ax=ax
            )
            ax.set_title('Feature Importance in Decision Tree', fontsize=14)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            st.pyplot(fig)
            plt.close()
                        
            st.divider()
            
            # Detailed metrics
            st.markdown("### 📋 Detailed Classification Report")
            report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

##############################################################sidebar#############################################################################

# Educational sidebar content
st.sidebar.divider()
st.sidebar.markdown("### FAQs")
with st.sidebar.expander("What is overfitting?"):
    st.markdown("""
    **Overfitting** occurs when a model learns the training data too well, including its noise and outliers. This results in:
    - Very high training accuracy
    - Lower test accuracy
    - Poor generalization to new data \n
    You can use exams as an analogy to understand overfitting. 
    Basically, overfitting is when you memorize the answers to the practice problems, only to find out your professor didnt use any of them on the final exam.
    
    **Solution:** Reduce model complexity (lower max_depth, increase min_samples_split)
    """)

with st.sidebar.expander("What is underfitting?"):
    st.markdown("""
    **Underfitting** occurs when a model is too simple to capture the underlying 
    patterns in the data. This results in:
    1: Low training accuracy
    2: Low test accuracy
    3: Poor performance overall
    
    **Solution:** Increase model complexity (higher max_depth, more features)
    """)