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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperparameters import hyperparameters

#################################################### initialisation #############################################################################################

st.set_page_config(page_title="Machine Learning interactive website", layout="wide")

def customFonts():
    st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        padding: 2rem 1rem !important;
    }
    
    /* ============= SIDEBAR TEXT COLORS - BRIGHT YELLOW ============= */
    
    /* Title */
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #FFEF00 !important;  /* Yellow */
        margin-bottom: 1.5rem !important;
        text-align: center !important;
        text-shadow: 0 0 20px rgba(255, 239, 0, 0.3) !important;
    }
    
    /* Section headers (h3) */
    [data-testid="stSidebar"] h3 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #e94560 !important;  /* Keep red for contrast */
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid rgba(233, 69, 96, 0.3) !important;
    }
    
    /* All general text, labels */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #FFEF00 !important;  /* Yellow */
        line-height: 1.6 !important;
    }
    
    /* Selectbox label */
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #FFEF00 !important;  /* Yellow */
        margin-bottom: 0.5rem !important;
    }
    
    /* Selectbox selected value */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 239, 0, 0.2) !important;
        border-radius: 8px !important;
        color: #FFEF00 !important;  /* Yellow */
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: rgba(255, 239, 0, 0.5) !important;
        box-shadow: 0 0 15px rgba(255, 239, 0, 0.2) !important;
    }
    
    /* Slider label */
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #FFEF00 !important;  /* Yellow */
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        margin-top: 1rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Slider track */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, rgba(255, 239, 0, 0.2) 0%, rgba(233, 69, 96, 0.2) 100%) !important;
    }
    
    /* Slider thumb */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #FFEF00 !important;  /* Yellow */
        box-shadow: 0 0 15px rgba(255, 239, 0, 0.5) !important;
    }
    
    /* Slider value */
    [data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
        color: #FFEF00 !important;  /* Yellow */
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }
    
    /* Help tooltip icon */
    [data-testid="stSidebar"] .stTooltipIcon {
        color: rgba(255, 239, 0, 0.6) !important;
    }
    
    /* Divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(233, 69, 96, 0.3) !important;
        margin: 2rem 0 !important;
    }
    
    /* Expander header */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(233, 69, 96, 0.1) !important;
        border: 1px solid rgba(233, 69, 96, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #e94560 !important;  /* Keep red for contrast */
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(233, 69, 96, 0.2) !important;
        border-color: rgba(233, 69, 96, 0.5) !important;
        box-shadow: 0 0 15px rgba(233, 69, 96, 0.2) !important;
    }
    
    /* Expander content */
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 239, 0, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent p {
        color: #FFEF00 !important;  /* Yellow */
    }
    
    /* Metric value */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #FFEF00 !important;  /* Yellow */
        text-shadow: 0 0 10px rgba(255, 239, 0, 0.3) !important;
    }
    
    /* Metric label */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #FFEF00 !important;  /* Yellow */
        font-weight: 600 !important;
    }
    
    /* Train Model button */
    [data-testid="stSidebar"] .stButton button {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        background: linear-gradient(135deg, #e94560 0%, #d63447 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        width: 100% !important;
        margin-top: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #d63447 0%, #e94560 100%) !important;
        box-shadow: 0 6px 25px rgba(233, 69, 96, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    
    /* FAQ section title - keep cyan for variety */
    [data-testid="stSidebar"] h3:last-of-type {
        color: #66fcf1 !important;
        border-bottom-color: rgba(102, 252, 241, 0.3) !important;
    }
    
    /* Caption text */
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 239, 0, 0.7) !important;  /* Semi-transparent yellow */
        font-size: 0.85rem !important;
        font-style: italic !important;
    }
                
        /* AGGRESSIVE FIX: Force all h3 in sidebar to be yellow */
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] div h3 {
        color: #FFEF00 !important;
        border-bottom-color: rgba(255, 239, 0, 0.3) !important;
    }
                
    </style>
    """, unsafe_allow_html=True)

customFonts()

# st.title("Machine Learning interactive website")
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

def goToDatasetInfo():
    st.session_state.page = 'dataset_info'

def goToProblemChoice():
    st.session_state.page = 'problem_choice'

def goToModels(problem_type):
    st.session_state.problem_type = problem_type
    st.session_state.page = 'models'

##########################################################welcome page############################################################################################
if st.session_state.page == 'welcome':
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            text-align: center;
            font-size: 3.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            margin-top: 2rem;
        }
        
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            margin-bottom: 3rem;
            color: #66fcf1;
        }
        
        /* Flip card container */
        .card {
            width: 100%;
            height: 500px;
            margin: 1rem auto;
            perspective: 1000px;
            cursor: pointer;
        }
        
        .card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            transition: transform 0.6s;
            transform-style: preserve-3d;
        }
        
        .card:hover .card-inner {
            transform: rotateY(180deg);
        }
        
        .card-front,
        .card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .card-front {
            background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        }
        
        .card-back {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: rotateY(180deg);
            padding: 2.5rem 2rem;
        }
        
        /* Front card styling */
        .card-front-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .card-front-subtitle {
            font-size: 1rem;
            color: #666;
            text-align: center;
        }
        
        /* Back card styling */
        .card-back-title {
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .mini-items {
            width: 100%;
        }
        
        .mini-item {
            background: rgba(255, 255, 255, 0.15);
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin-bottom: 0.8rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: left;
        }
        
        .mini-icon {
            font-size: 1.2rem;
            margin-right: 0.8rem;
        }
        
        .mini-text {
            font-size: 0.95rem;
            line-height: 1.4;
        }
        
        /* Alternative gradient for second card */
        .card-back-alt {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and subtitle
    st.markdown('<h1 class="main-title">Master Machine Learning With Real Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Learn Machine Learning Through Interactive Exploration</p>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns(2, gap="large")
    
    # Left card - What You Can Do
    with col1:
        st.markdown("""
            <div class="card">
                <div class="card-inner">
                    <div class="card-front">
                        <div class="card-front-title">What You Can Do</div>
                        <div class="card-front-subtitle">Hover to explore →</div>
                    </div>
                    <div class="card-back">
                        <div class="mini-items">
                            <div class="mini-item">
                                <span class="mini-text"><strong>Adjust hyperparameters</strong> and see immediate effects</span>
                            </div>
                            <div class="mini-item">
                                <span class="mini-text"><strong>Compare accuracy</strong> to understand overfitting</span>
                            </div>
                            <div class="mini-item">
                                <span class="mini-text"><strong>Visualize performance</strong> through confusion matrices</span>
                            </div>
                            <div class="mini-item">
                                <span class="mini-text"><strong>Discover optimal settings</strong> by experimenting</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card">
                <div class="card-inner">
                    <div class="card-front">
                        <div class="card-front-title">What You'll Learn</div>
                        <div class="card-front-subtitle">Hover to explore →</div>
                    </div>
                    <div class="card-back card-back-alt">
                        <div class="mini-items">
                            <div class="mini-item">
                                <span class="mini-text"><strong>Hyperparameter effects</strong> on model performance</span>
                            </div>
                            <div class="mini-item">
                                <span class="mini-text"><strong>Overfitting vs. underfitting</strong> concepts</span>
                            </div>
                            <div class="mini-item">
                                <span class="mini-text"><strong>Interpret confusion matrices</strong> and metrics</span>
                            </div>
                            <div class="mini-item">
                                <span class="mini-text"><strong>Model selection</strong> for different scenarios</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("Get Started →", on_click=goToDatasetInfo, type="primary", use_container_width=True)

######################################################## dataset info page ###################################################################################
elif st.session_state.page == 'dataset_info':
    st.title("ⓘ About the dataset")
    
    st.markdown("""
    ## The RMS Titanic Crash
    
    The RMS Titanic was a British ocean liner that sank in the early hours of 15 April 1912 as a result of striking an iceberg on her maiden voyage from Southampton, England, to New York City, United States. 
    Of the estimated 2,224 passengers and crew aboard, approximately 1,500 died, making the incident one of the deadliest peacetime sinkings of a single ship.
    
    ### The Challenge
    
    The challenge is to predict whether a passenger survived or not based on certain data.
    This is a classification problem, and will have a binary outcome: Survived(1) or Dead(0).
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Dataset Features
        
        **Passenger Information:**
        - **Pclass:** Passenger class (1st/2nd/3rd, higher the class, richer the person and more expensive the fare)
        - **Sex:** Male/Female  
        - **Age:** In years
        - **SibSp:** Number of siblings/spouses onboard
        - **Parch:** Number of parents/children onboard
        
        **Ticket Information:**
        - **Fare:** Ticket price paid
        - **Deck:** Deck location (ABC, DE, FG, M=Missing)
        - **Embarked:** Port of embarking
          - C = Cherbourg
          - Q = Queenstown  
          - S = Southampton
        """)
    
    with col2:
        st.markdown("""
        ### Important stats
        
        - **Total Passengers:** 891 (in training set)
        - **Survival Rate:** ~38%
        - **Gender Distribution:** ~65% male, ~35% female
        - **Class Distribution:**
          - 1st Class: ~24%
          - 2nd Class: ~21%
          - 3rd Class: ~55%
        
        ### Interesting patterns
        
        - Women had much higher survival rates than men
        - Higher class passengers had better survival chances
        - Children were more likely to survive
        - Fare correlates with survival (wealthier passengers had higher survival chances)
        """)
    
    st.divider()
    
    # st.info("💡 **Did you know?** The famous \"women and children first\" protocol significantly influenced survival rates!")
    
    st.button("Continue to Problem Selection →", on_click=goToProblemChoice, type="primary", use_container_width=True)

################################################# model selection ############################################################################################
elif st.session_state.page == 'problem_choice':
    st.title("🕮 Choose Your Learning Path")
    
    st.markdown("""
    Machine learning problems fall into different categories. For this dataset, we'll focus on:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Classification
        
        **Goal:** Predict discrete categories or classes
        
        **For Titanic:** Predict whether a passenger survived (1) or died (0)
        
        **Models:**
        - Decision trees
        - Random forest
        - Logistic regression
        - SVM(Support vector machines)
        - KNN(K Nearest neighbours)
        
        **Learning outcomes:**
        - Binary classification
        - Overfitting vs underfitting
        - Confusion matrices
        - Precision, recall, F1 score
        """)
        
        st.button(
            "Select →", 
            on_click=lambda: goToModels('classification'),
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.markdown("""
        ### Regression
        
        **Goal:** Predict continuous numerical values
        
        **Example:** Predicting a passenger's age or ticket fare
        
        **Status:** Coming Soon!
        
        **Models:**
        - Linear regression
        - Ridge regression
        - Lasso regression
        - Decision tree regressor
        - Random forest regressor
        """)
        
        st.button(
            "Regression (Coming Soon)", 
            disabled=True,
            use_container_width=True
        )

# ==================== PAGE 4: MODELS INTERFACE ====================
elif st.session_state.page == 'models':
    # Load data (cached)
    @st.cache_data
    def get_data():
        return preprocessingData()
    
    X_train, X_test, y_train, y_test, scaler, features = get_data()
    
    # Title with back button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🚢 Titanic Survival Prediction - Classification Models")
    with col2:
        if st.button("← Back to Start"):
            st.session_state.page = 'welcome'
            st.rerun()
    
    st.markdown("Experiment with different models and hyperparameters to maximize prediction accuracy!")
    
    # Display dataset statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", len(X_train) + len(X_test))
    col2.metric("Training Samples", len(X_train))
    col3.metric("Test Samples", len(X_test))
    
    st.divider()
    
    # Sidebar for model selection (ONLY classification models)
    st.sidebar.title("🎛️ Model Configuration")
    
    if st.session_state.problem_type == 'classification':
        model_choice = st.sidebar.selectbox(
            "Select Classification Model",
            ["Decision Tree", "Random Forest", "Logistic Regression", "SVM", "KNN"]
        )
    else:  # For future regression
        model_choice = st.sidebar.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Ridge Regression"]  # Placeholder
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

        st.sidebar.divider()
        with st.sidebar.expander("🎯 Best Accuracy"):
            best_acc = hyperparameters['Decision Tree']['best_accuracy']
            st.metric("Best Test Accuracy", f"{best_acc:.2%}")
            st.caption("Maximum accuracy from GridSearchCV")
        
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
        st.sidebar.markdown("### Hyperparameters")
        
        # Hyperparameter controls
        C = st.sidebar.slider(
            "Regularization Strength",
            min_value=0.001,
            max_value=2.0,
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
                    with st.expander("ℹ️ Understanding the Confusion Matrix"):
                        st.markdown("""
                        - **Top-left (True Negative):** Correctly predicted deaths
                        - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                        - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                        - **Bottom-right (True Positive):** Correctly predicted survival
                        
                        A good model has high numbers on the diagonal (top-left and bottom-right).
                        """)
                    
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
    elif model_choice == "SVM":
        st.sidebar.markdown("### Hyperparameters")
        
        # Hyperparameter controls
        C = st.sidebar.slider(
            "C (Regularization Parameter)",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            help="Penalty parameter. Smaller C means stronger regularization (wider margin, more misclassifications allowed)."
        )
        
        kernel = st.sidebar.selectbox(
            "Kernel",
            ["linear", "rbf", "poly", "sigmoid"],
            index=1,  # rbf is default
            help="Kernel function. 'rbf' (Radial Basis Function) works well for most cases. 'linear' for linearly separable data."
        )
        
        gamma = st.sidebar.selectbox(
            "Gamma",
            ["scale", "auto"],
            help="Kernel coefficient. 'scale' is recommended for most cases."
        )
        
        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training SVM"):
                svmModel = SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    random_state=42
                )
                svmModel.fit(X_train, y_train)
                
                yTrainPred = svmModel.predict(X_train)
                yTestPred = svmModel.predict(X_test)
                
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
                with st.expander("ℹ️ Understanding the Confusion Matrix"):
                        st.markdown("""
                        - **Top-left (True Negative):** Correctly predicted deaths
                        - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                        - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                        - **Bottom-right (True Positive):** Correctly predicted survival
                        
                        A good model has high numbers on the diagonal (top-left and bottom-right).
                        """)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Oranges',
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
                
                # Detailed metrics
                st.markdown("### Detailed Classification Report")
                report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)


    ######################################################### KNN ########################################################################################   
    elif model_choice == "KNN":
        st.sidebar.markdown("### Hyperparameters")
        n_neighbors = st.sidebar.slider(
            "Number of Neighbors (K)",
            min_value=1,
            max_value=30,
            value=5,  
            help="Number of neighbors to consider. Smaller K = more complex decision boundary (may overfit). Larger K = smoother boundary (may underfit)."
        )
        
        weights = st.sidebar.selectbox(
            "Weights",
            ["uniform", "distance"],
            help="'uniform': All neighbors have equal weight. 'distance': Closer neighbors have more influence."
        )
        
        metric = st.sidebar.selectbox(
            "Distance Metric",
            ["euclidean", "manhattan", "minkowski"],
            help="Method to calculate distance between points. 'euclidean' is most common; 'manhattan' can work better in high dimensions."
        )
        
        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training KNN"):
                knnModel = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric
                )
                knnModel.fit(X_train, y_train)
                
                yTrainPred = knnModel.predict(X_train)
                yTestPred = knnModel.predict(X_test)

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
                
                st.markdown("### 🔢 Confusion Matrix (Test Set)")
                with st.expander("ℹ️ Understanding the Confusion Matrix"):
                        st.markdown("""
                        - **Top-left (True Negative):** Correctly predicted deaths
                        - **Top-right (False Positive):** Incorrectly predicted survival (actually died)
                        - **Bottom-left (False Negative):** Incorrectly predicted death (actually survived)
                        - **Bottom-right (True Positive):** Correctly predicted survival
                        
                        A good model has high numbers on the diagonal (top-left and bottom-right).
                        """)

                
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='YlOrBr',
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
                
                # Detailed metrics
                st.markdown("### Detailed Classification Report")
                report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

    ##############################################################sidebar#############################################################################

    # Educational sidebar content
    st.sidebar.divider()
    st.sidebar.markdown("### FAQs")
    with st.sidebar.expander("💭 What is overfitting?"):
        st.markdown("""
        **Overfitting** occurs when a model learns the training data too well, including its noise and outliers. This results in:
        - Very high training accuracy
        - Lower test accuracy
        - Poor generalization to new data
        
        Think of it like memorizing exam answers without understanding the concepts.
        
        **Solution:** Reduce model complexity
        """)

    with st.sidebar.expander("💭 What is underfitting?"):
        st.markdown("""
        **Underfitting** occurs when a model is too simple to capture the underlying patterns in the data. This results in:
        - Low training accuracy
        - Low test accuracy
        - Poor performance overall
        
        **Solution:** Increase model complexity
        """)