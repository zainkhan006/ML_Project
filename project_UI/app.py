import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocessingData
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
if 'engineered_features' not in st.session_state:
    st.session_state.engineered_features = []  # List of {name, formula, values_train, values_test}
if 'ensemble_results' not in st.session_state:
    st.session_state.ensemble_results = []  # List of {name, method, models, train_accuracy, test_accuracy}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}  # {name: {model, train_accuracy, test_accuracy, hyperparams, problem_type}}
if 'last_visualization' not in st.session_state:
    st.session_state.last_visualization = None  # {model_type, model, X_train, y_train, features, hyperparams}

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
            color: #FFEF00;
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
        
        /* CYBERPUNK FRONT CARD - Simple gradient, no pseudo-elements */
        .card-front {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        }
        
        /* CYBERPUNK BACK CARD - Purple gradient */
        .card-back {
            background: linear-gradient(135deg, #8338ec 0%, #3a0ca3 100%);
            color: white;
            transform: rotateY(180deg);
            padding: 2.5rem 2rem;
        }
        
        /* Front card styling */
        .card-front-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #66fcf1;
            margin-bottom: 1rem;
            text-align: center;
            text-shadow: 0 0 20px rgba(102, 252, 241, 0.5);
        }
        
        .card-front-subtitle {
            font-size: 1rem;
            color: #c9d1d9;
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
            transition: all 0.3s ease;
        }
        
        .mini-item:hover {
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(102, 252, 241, 0.5);
            transform: translateX(5px);
        }
        
        .mini-icon {
            font-size: 1.2rem;
            margin-right: 0.8rem;
        }
        
        .mini-text {
            font-size: 0.95rem;
            line-height: 1.4;
        }
        
        .mini-text strong {
            color: #66fcf1;
        }
        
        /* Alternative gradient for second card - Hot pink */
        .card-back-alt {
            background: linear-gradient(135deg, #ff006e 0%, #fb5607 100%);
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
        - Perceptrons
        - Neural Networks
        
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
            use_container_width=True,
            key="classification_select"
        )

    with col2:
        st.markdown("""
        ### Regression

        **Goal:** Predict continuous numerical values

        **Example:** Predicting a passenger's age or ticket fare

        **Status:** Coming soon

        **Models:**
        - Linear regression
        - Ridge regression
        - Lasso regression

        **What you'll learn:**
        - Continuous value prediction
        - Mean squared error and R² score
        - L1 and L2 regularisation techniques
        - Feature scaling and it's importance
        - Handling outliers in predictions
        """)

        st.button(
            "Select →",
            on_click=lambda: goToModels('regression'),
            type="primary",
            use_container_width=True,
            key="regression_select"
        )

############################################################## FEATURE ENGINEERING DIALOG ##############################################################################
@st.dialog("Feature Engineering", width="large")
def featureEngineeringDialog():
    # Load data for feature engineering
    X_train, X_test, y_train, y_test, scaler, baseFeatures = preprocessingData()

    # Convert to DataFrame for easier manipulation
    X_train_df = pd.DataFrame(X_train, columns=baseFeatures)
    X_test_df = pd.DataFrame(X_test, columns=baseFeatures)

    st.markdown("Create new features to improve model accuracy")

    # Main layout: two columns
    leftCol, rightCol = st.columns([1, 2])

    with leftCol:
        st.markdown("### Current Features")

        # Show base features
        st.markdown("**Base Features:**")
        for feature in baseFeatures:
            st.markdown(f"- {feature}")

        # Show engineered features
        if st.session_state.engineered_features:
            st.markdown("**Engineered Features:**")
            for i, engFeature in enumerate(st.session_state.engineered_features):
                featureCol1, featureCol2 = st.columns([3, 1])
                with featureCol1:
                    st.markdown(f"✨ **{engFeature['name']}**")
                    st.caption(f"Formula: {engFeature['formula']}")
                with featureCol2:
                    if st.button("🗑️", key=f"dialog_remove_{i}", help="Remove this feature"):
                        st.session_state.engineered_features.pop(i)
                        st.rerun()

        st.divider()

        # Feature count summary
        totalFeatures = len(baseFeatures) + len(st.session_state.engineered_features)
        st.metric("Total Features", totalFeatures)

    with rightCol:
        st.markdown("### Create New Feature")

        # Tabs for different operation types
        tab1, tab2, tab3, tab4 = st.tabs(["Arithmetic", "Presets", "Transform", "Polynomial"])

        allFeatures = baseFeatures + [f['name'] for f in st.session_state.engineered_features]

        ############################################################## Arithmetic Tab ##############################################################################
        with tab1:
            with st.popover("ℹ️ What is Arithmetic?"):
                st.markdown("**Arithmetic Operations**")
                st.write("Combine two features using basic math operations (+, -, ×, ÷). This can reveal relationships between features.")
                st.write("**Example:** Age × Pclass might show that older passengers in lower classes had different survival rates.")

            arithmeticCol1, arithmeticCol2, arithmeticCol3 = st.columns(3)

            with arithmeticCol1:
                feature1 = st.selectbox("Feature 1", allFeatures, key="dialog_arith_f1")
            with arithmeticCol2:
                operation = st.selectbox("Operation", ["Add (+)", "Subtract (-)", "Multiply (×)", "Divide (÷)"], key="dialog_arith_op")
            with arithmeticCol3:
                feature2 = st.selectbox("Feature 2", allFeatures, key="dialog_arith_f2")

            # Auto-generate name
            opSymbols = {"Add (+)": "+", "Subtract (-)": "-", "Multiply (×)": "×", "Divide (÷)": "÷"}
            defaultName = f"{feature1}_{opSymbols[operation]}_{feature2}"
            arithmeticName = st.text_input("Feature Name", value=defaultName, key="dialog_arith_name")

            if st.button("Create Feature", key="dialog_create_arith", type="primary"):
                # Get values
                if feature1 in baseFeatures:
                    vals1_train = X_train_df[feature1].values
                    vals1_test = X_test_df[feature1].values
                else:
                    engF = next(f for f in st.session_state.engineered_features if f['name'] == feature1)
                    vals1_train = engF['values_train']
                    vals1_test = engF['values_test']

                if feature2 in baseFeatures:
                    vals2_train = X_train_df[feature2].values
                    vals2_test = X_test_df[feature2].values
                else:
                    engF = next(f for f in st.session_state.engineered_features if f['name'] == feature2)
                    vals2_train = engF['values_train']
                    vals2_test = engF['values_test']

                # Apply operation
                if operation == "Add (+)":
                    newValsTrain = vals1_train + vals2_train
                    newValsTest = vals1_test + vals2_test
                elif operation == "Subtract (-)":
                    newValsTrain = vals1_train - vals2_train
                    newValsTest = vals1_test - vals2_test
                elif operation == "Multiply (×)":
                    newValsTrain = vals1_train * vals2_train
                    newValsTest = vals1_test * vals2_test
                else:  # Divide
                    newValsTrain = vals1_train / (vals2_train + 0.001)
                    newValsTest = vals1_test / (vals2_test + 0.001)

                # Add to session state
                st.session_state.engineered_features.append({
                    'name': arithmeticName,
                    'formula': f"{feature1} {opSymbols[operation]} {feature2}",
                    'values_train': newValsTrain,
                    'values_test': newValsTest
                })
                st.success(f"Created feature: {arithmeticName}")
                st.rerun()

        ############################################################## Presets Tab ##############################################################################
        with tab2:
            with st.popover("ℹ️ What are Presets?"):
                st.markdown("**Preset Feature Combinations**")
                st.write("These are commonly used feature combinations for the Titanic dataset that have been proven to improve model performance.")
                st.write("**Why they work:** Domain knowledge about the Titanic disaster suggests these combinations capture important survival factors.")

            presets = {
                'FamilySize': {
                    'formula': 'SibSp + Parch + 1',
                    'description': 'Total family members including the passenger. Larger families may have had different survival rates.',
                    'calc_train': X_train_df['SibSp'].values + X_train_df['Parch'].values + 1,
                    'calc_test': X_test_df['SibSp'].values + X_test_df['Parch'].values + 1
                },
                'IsAlone': {
                    'formula': '1 if (SibSp + Parch) == 0 else 0',
                    'description': 'Whether passenger is traveling alone. Solo travelers may have had different survival chances.',
                    'calc_train': ((X_train_df['SibSp'].values + X_train_df['Parch'].values) == 0).astype(int),
                    'calc_test': ((X_test_df['SibSp'].values + X_test_df['Parch'].values) == 0).astype(int)
                },
                'AgeClass': {
                    'formula': 'Age × Pclass',
                    'description': 'Interaction between age and class. Young first-class passengers vs old third-class passengers.',
                    'calc_train': X_train_df['Age'].values * X_train_df['Pclass'].values,
                    'calc_test': X_test_df['Age'].values * X_test_df['Pclass'].values
                },
                'FarePerPerson': {
                    'formula': 'Fare ÷ (SibSp + Parch + 1)',
                    'description': 'Fare adjusted for family size. Shows actual spending power per person.',
                    'calc_train': X_train_df['Fare'].values / (X_train_df['SibSp'].values + X_train_df['Parch'].values + 1),
                    'calc_test': X_test_df['Fare'].values / (X_test_df['SibSp'].values + X_test_df['Parch'].values + 1)
                }
            }

            # Check which presets are already created
            existingNames = [f['name'] for f in st.session_state.engineered_features]

            for presetName, presetInfo in presets.items():
                presetCol1, presetCol2 = st.columns([3, 1])
                with presetCol1:
                    st.markdown(f"**{presetName}**")
                    st.caption(f"Formula: {presetInfo['formula']}")
                    st.caption(presetInfo['description'])
                with presetCol2:
                    if presetName in existingNames:
                        st.success("Added ✓")
                    else:
                        if st.button("Add", key=f"dialog_preset_{presetName}"):
                            st.session_state.engineered_features.append({
                                'name': presetName,
                                'formula': presetInfo['formula'],
                                'values_train': presetInfo['calc_train'],
                                'values_test': presetInfo['calc_test']
                            })
                            st.rerun()
                st.divider()

        ############################################################## Transform Tab ##############################################################################
        with tab3:
            with st.popover("ℹ️ What are Transformations?"):
                st.markdown("**Feature Transformations**")
                st.write("Apply mathematical transformations to a single feature. This can help with:")
                st.write("- **Log:** Reduces impact of outliers, good for skewed distributions")
                st.write("- **Square Root:** Milder than log, also handles skew")
                st.write("- **Square:** Emphasizes larger values")
                st.write("- **Binning:** Groups continuous values into categories")

            transformFeature = st.selectbox("Select Feature to Transform", allFeatures, key="dialog_transform_feature")

            transformType = st.selectbox("Transformation Type", [
                "Log (log1p)",
                "Square Root",
                "Square",
                "Bin (3 groups)",
                "Bin (5 groups)"
            ], key="dialog_transform_type")

            # Auto-generate name
            transformSuffixes = {
                "Log (log1p)": "_log",
                "Square Root": "_sqrt",
                "Square": "_sq",
                "Bin (3 groups)": "_bin3",
                "Bin (5 groups)": "_bin5"
            }
            transformDefaultName = f"{transformFeature}{transformSuffixes[transformType]}"
            transformName = st.text_input("Feature Name", value=transformDefaultName, key="dialog_transform_name")

            if st.button("Create Feature", key="dialog_create_transform", type="primary"):
                # Get values
                if transformFeature in baseFeatures:
                    valsToTransformTrain = X_train_df[transformFeature].values
                    valsToTransformTest = X_test_df[transformFeature].values
                else:
                    engF = next(f for f in st.session_state.engineered_features if f['name'] == transformFeature)
                    valsToTransformTrain = engF['values_train']
                    valsToTransformTest = engF['values_test']

                # Apply transformation
                if transformType == "Log (log1p)":
                    newValsTrain = np.log1p(np.abs(valsToTransformTrain))
                    newValsTest = np.log1p(np.abs(valsToTransformTest))
                    formula = f"log(1 + |{transformFeature}|)"
                elif transformType == "Square Root":
                    newValsTrain = np.sqrt(np.abs(valsToTransformTrain))
                    newValsTest = np.sqrt(np.abs(valsToTransformTest))
                    formula = f"sqrt(|{transformFeature}|)"
                elif transformType == "Square":
                    newValsTrain = valsToTransformTrain ** 2
                    newValsTest = valsToTransformTest ** 2
                    formula = f"{transformFeature}²"
                elif transformType == "Bin (3 groups)":
                    try:
                        newValsTrain = pd.qcut(valsToTransformTrain, 3, labels=[0, 1, 2], duplicates='drop').astype(float).values
                        newValsTest = pd.cut(valsToTransformTest, bins=3, labels=[0, 1, 2]).astype(float).values
                    except:
                        newValsTrain = pd.cut(valsToTransformTrain, bins=3, labels=[0, 1, 2]).astype(float).values
                        newValsTest = pd.cut(valsToTransformTest, bins=3, labels=[0, 1, 2]).astype(float).values
                    formula = f"bin({transformFeature}, 3 groups)"
                else:  # Bin 5 groups
                    try:
                        newValsTrain = pd.qcut(valsToTransformTrain, 5, labels=[0, 1, 2, 3, 4], duplicates='drop').astype(float).values
                        newValsTest = pd.cut(valsToTransformTest, bins=5, labels=[0, 1, 2, 3, 4]).astype(float).values
                    except:
                        newValsTrain = pd.cut(valsToTransformTrain, bins=5, labels=[0, 1, 2, 3, 4]).astype(float).values
                        newValsTest = pd.cut(valsToTransformTest, bins=5, labels=[0, 1, 2, 3, 4]).astype(float).values
                    formula = f"bin({transformFeature}, 5 groups)"

                # Handle NaN values
                newValsTrain = np.nan_to_num(newValsTrain, nan=0.0)
                newValsTest = np.nan_to_num(newValsTest, nan=0.0)

                st.session_state.engineered_features.append({
                    'name': transformName,
                    'formula': formula,
                    'values_train': newValsTrain,
                    'values_test': newValsTest
                })
                st.success(f"Created feature: {transformName}")
                st.rerun()

        ############################################################## Polynomial Tab ##############################################################################
        with tab4:
            with st.popover("ℹ️ What are Polynomial Features?"):
                st.markdown("**Polynomial Features**")
                st.write("Creates polynomial combinations of selected features. For two features A and B, this generates:")
                st.write("- A² (A squared)")
                st.write("- B² (B squared)")
                st.write("- A×B (interaction term)")
                st.write("**When to use:** When you suspect non-linear relationships between features and the target.")

            polyCol1, polyCol2 = st.columns(2)
            with polyCol1:
                polyFeature1 = st.selectbox("Feature 1", allFeatures, key="dialog_poly_f1")
            with polyCol2:
                polyFeature2 = st.selectbox("Feature 2", allFeatures, key="dialog_poly_f2", index=min(1, len(allFeatures)-1))

            if st.button("Generate Polynomial Features", key="dialog_create_poly", type="primary"):
                # Get values
                if polyFeature1 in baseFeatures:
                    polyVals1Train = X_train_df[polyFeature1].values
                    polyVals1Test = X_test_df[polyFeature1].values
                else:
                    engF = next(f for f in st.session_state.engineered_features if f['name'] == polyFeature1)
                    polyVals1Train = engF['values_train']
                    polyVals1Test = engF['values_test']

                if polyFeature2 in baseFeatures:
                    polyVals2Train = X_train_df[polyFeature2].values
                    polyVals2Test = X_test_df[polyFeature2].values
                else:
                    engF = next(f for f in st.session_state.engineered_features if f['name'] == polyFeature2)
                    polyVals2Train = engF['values_train']
                    polyVals2Test = engF['values_test']

                # Create polynomial features
                polyFeaturesList = [
                    {
                        'name': f"{polyFeature1}_squared",
                        'formula': f"{polyFeature1}²",
                        'values_train': polyVals1Train ** 2,
                        'values_test': polyVals1Test ** 2
                    },
                    {
                        'name': f"{polyFeature2}_squared",
                        'formula': f"{polyFeature2}²",
                        'values_train': polyVals2Train ** 2,
                        'values_test': polyVals2Test ** 2
                    },
                    {
                        'name': f"{polyFeature1}_{polyFeature2}_interaction",
                        'formula': f"{polyFeature1} × {polyFeature2}",
                        'values_train': polyVals1Train * polyVals2Train,
                        'values_test': polyVals1Test * polyVals2Test
                    }
                ]

                # Add only features that don't already exist
                existingNames = [f['name'] for f in st.session_state.engineered_features]
                addedCount = 0
                for polyF in polyFeaturesList:
                    if polyF['name'] not in existingNames:
                        st.session_state.engineered_features.append(polyF)
                        addedCount += 1

                if addedCount > 0:
                    st.success(f"Added {addedCount} polynomial features!")
                    st.rerun()
                else:
                    st.info("All polynomial features already exist.")

    st.divider()

    # Preview section
    if st.session_state.engineered_features:
        st.markdown("### Feature Preview")
        previewData = X_train_df.copy()
        for engFeature in st.session_state.engineered_features:
            previewData[engFeature['name']] = engFeature['values_train']
        st.dataframe(previewData.head(5), use_container_width=True)

    # Done button
    if st.button("Done", type="primary", use_container_width=True):
        st.rerun()

############################################################## MODEL FACTORY FUNCTIONS ##############################################################################
def getClassificationModel(name):
    """Returns a classification model with default hyperparameters"""
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }
    return models.get(name)

def getRegressionModel(name):
    """Returns a regression model with default hyperparameters"""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=5000)
    }
    return models.get(name)

############################################################## TRAINING VISUALIZATION DIALOG ##############################################################################
@st.dialog("Training Visualization", width="large")
def trainingVisualizationDialog():
    """Shows animated training visualization for the last trained model"""
    from training_visualizations import (
        visualize_decision_tree, visualize_random_forest, visualize_logistic_regression,
        visualize_svm, visualize_knn, visualize_perceptron, visualize_neural_network,
        visualize_linear_regression, visualize_ridge_regression, visualize_lasso_regression
    )

    vizData = st.session_state.last_visualization

    if vizData is None:
        st.warning("No model has been trained yet. Train a model first to see the visualization.")
        return

    modelType = vizData['model_type']
    model = vizData['model']
    X_train = vizData['X_train']
    y_train = vizData['y_train']
    features = vizData['features']
    hyperparams = vizData['hyperparams']

    st.markdown(f"### {modelType} Training Animation")
    st.caption("Watch how the model learns from the training data step by step")

    # Call appropriate visualization based on model type
    if modelType == "Decision Tree":
        visualize_decision_tree(model, X_train, y_train, features, hyperparams['max_depth'])
    elif modelType == "Random Forest":
        visualize_random_forest(model, X_train, y_train, features, hyperparams['n_estimators'], hyperparams['max_depth'])
    elif modelType == "Logistic Regression":
        visualize_logistic_regression(model, X_train, y_train, features, hyperparams['C'])
    elif modelType == "SVM":
        visualize_svm(model, X_train, y_train, features, hyperparams['C'], hyperparams['kernel'])
    elif modelType == "KNN":
        visualize_knn(model, X_train, y_train, features, hyperparams['n_neighbors'])
    elif modelType == "Perceptron":
        visualize_perceptron(model, X_train, y_train, features, hyperparams['max_iter'], hyperparams['eta0'])
    elif modelType == "Neural Network":
        visualize_neural_network(model, X_train, y_train, hyperparams['hidden_layer_sizes'], hyperparams['activation'], hyperparams['max_iter'])
    elif modelType == "Linear Regression":
        visualize_linear_regression(model, X_train, y_train, features)
    elif modelType == "Ridge Regression":
        visualize_ridge_regression(model, X_train, y_train, features, hyperparams['alpha'])
    elif modelType == "Lasso Regression":
        visualize_lasso_regression(model, X_train, y_train, features, hyperparams['alpha'])

############################################################## ENSEMBLE LEARNING DIALOG ##############################################################################
@st.dialog("Ensemble Learning", width="large")
def ensembleLearningDialog():
    # Load data
    X_train_base, X_test_base, y_train, y_test, scaler, baseFeatures = preprocessingData()

    # Convert to DataFrame and add engineered features
    X_train_df = pd.DataFrame(X_train_base, columns=baseFeatures)
    X_test_df = pd.DataFrame(X_test_base, columns=baseFeatures)

    for engFeature in st.session_state.engineered_features:
        X_train_df[engFeature['name']] = engFeature['values_train']
        X_test_df[engFeature['name']] = engFeature['values_test']

    X_train = X_train_df.values
    X_test = X_test_df.values

    st.markdown("Combine your trained models into a powerful ensemble")

    # Filter saved models by current problem type
    availableModels = {
        name: info for name, info in st.session_state.trained_models.items()
        if info['problem_type'] == st.session_state.problem_type
    }

    if not availableModels:
        st.warning("No trained models yet! Train some models first using the sidebar, then come back to create an ensemble.")
        st.info("Each time you train a model, it gets saved automatically. You can then combine multiple saved models here.")
        if st.button("Close", type="primary", use_container_width=True):
            st.rerun()
        return

    # Two-column layout
    leftCol, rightCol = st.columns([1, 2])

    with leftCol:
        st.markdown("### Saved Models")
        st.caption("Select 2+ models to combine")

        selectedModelNames = []
        for modelName, modelInfo in availableModels.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.checkbox(modelName, key=f"ensemble_select_{modelName}"):
                    selectedModelNames.append(modelName)
            with col2:
                st.caption(f"{modelInfo['test_accuracy']:.1%}")

        st.divider()
        st.metric("Selected", f"{len(selectedModelNames)} models")

        if len(selectedModelNames) < 2:
            st.warning("Select at least 2 models")

        # Clear saved models option
        with st.expander("Manage Saved Models"):
            if st.button("Clear All Saved Models", type="secondary"):
                st.session_state.trained_models = {}
                st.rerun()

    with rightCol:
        st.markdown("### Configure Ensemble")

        # Ensemble method selection based on problem type
        if st.session_state.problem_type == 'classification':
            method = st.selectbox("Ensemble Method", [
                "Voting (Hard)", "Voting (Soft)", "Stacking", "Bagging", "AdaBoost"
            ], key="ensemble_method")
        else:
            method = st.selectbox("Ensemble Method", [
                "Voting (Average)", "Stacking", "Bagging", "AdaBoost"
            ], key="ensemble_method")

        # Info popover explaining each method
        with st.popover("ℹ️ What is this method?"):
            if "Voting" in method:
                st.markdown("**Voting Ensemble**")
                st.write("Each model makes a prediction, and the final result is determined by majority vote (hard) or averaged probabilities (soft).")
                st.write("**Best for:** Combining diverse models that make different types of errors.")
            elif method == "Stacking":
                st.markdown("**Stacking Ensemble**")
                st.write("A meta-model learns to combine base model predictions. Uses clones of your saved models.")
                st.write("**Best for:** When you want the ensemble to learn optimal model weights.")
            elif method == "Bagging":
                st.markdown("**Bagging Ensemble**")
                st.write("Trains multiple copies of the first selected model on random data subsets. Reduces variance.")
                st.write("**Best for:** Reducing overfitting of unstable models like Decision Trees.")
            elif method == "AdaBoost":
                st.markdown("**AdaBoost Ensemble**")
                st.write("Sequentially trains models, focusing on mistakes from previous models.")
                st.write("**Best for:** Boosting weak learners into strong ones.")

        # Custom name input
        defaultName = f"Ensemble_{len(st.session_state.ensemble_results) + 1}"
        ensembleName = st.text_input("Ensemble Name", value=defaultName, key="ensemble_name")

        st.divider()

        # Train button
        trainDisabled = len(selectedModelNames) < 2
        if st.button("Train Ensemble", type="primary", disabled=trainDisabled, use_container_width=True):
            with st.spinner(f"Training {method} ensemble with your saved models..."):
                try:
                    # Get trained model objects
                    from sklearn.base import clone

                    if st.session_state.problem_type == 'classification':
                        # Build estimators list using saved models (cloned for refitting)
                        estimators = [
                            (name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""),
                             clone(st.session_state.trained_models[name]['model']))
                            for name in selectedModelNames
                        ]

                        # Create ensemble based on method
                        if method == "Voting (Hard)":
                            ensemble = VotingClassifier(estimators=estimators, voting='hard')
                        elif method == "Voting (Soft)":
                            ensemble = VotingClassifier(estimators=estimators, voting='soft')
                        elif method == "Stacking":
                            ensemble = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
                        elif method == "Bagging":
                            baseModel = clone(st.session_state.trained_models[selectedModelNames[0]]['model'])
                            ensemble = BaggingClassifier(estimator=baseModel, n_estimators=10, random_state=42)
                        elif method == "AdaBoost":
                            baseModel = clone(st.session_state.trained_models[selectedModelNames[0]]['model'])
                            ensemble = AdaBoostClassifier(estimator=baseModel, n_estimators=50, random_state=42, algorithm='SAMME')

                        # Train
                        ensemble.fit(X_train, y_train)

                        # Evaluate
                        trainAcc = accuracy_score(y_train, ensemble.predict(X_train))
                        testAcc = accuracy_score(y_test, ensemble.predict(X_test))

                        # Store results
                        st.session_state.ensemble_results.append({
                            'name': ensembleName,
                            'method': method,
                            'models': selectedModelNames.copy(),
                            'train_accuracy': trainAcc,
                            'test_accuracy': testAcc,
                            'problem_type': 'classification'
                        })

                    else:  # Regression
                        estimators = [
                            (name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""),
                             clone(st.session_state.trained_models[name]['model']))
                            for name in selectedModelNames
                        ]

                        if method == "Voting (Average)":
                            ensemble = VotingRegressor(estimators=estimators)
                        elif method == "Stacking":
                            ensemble = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3)
                        elif method == "Bagging":
                            baseModel = clone(st.session_state.trained_models[selectedModelNames[0]]['model'])
                            ensemble = BaggingRegressor(estimator=baseModel, n_estimators=10, random_state=42)
                        elif method == "AdaBoost":
                            baseModel = clone(st.session_state.trained_models[selectedModelNames[0]]['model'])
                            ensemble = AdaBoostRegressor(estimator=baseModel, n_estimators=50, random_state=42)

                        # Train
                        ensemble.fit(X_train, y_train)

                        # Evaluate with threshold for classification accuracy
                        yTrainPred = ensemble.predict(X_train)
                        yTestPred = ensemble.predict(X_test)

                        yTrainPredBinary = (yTrainPred >= 0.5).astype(int)
                        yTestPredBinary = (yTestPred >= 0.5).astype(int)

                        trainAcc = accuracy_score(y_train, yTrainPredBinary)
                        testAcc = accuracy_score(y_test, yTestPredBinary)

                        r2Train = r2_score(y_train, yTrainPred)
                        r2Test = r2_score(y_test, yTestPred)

                        st.session_state.ensemble_results.append({
                            'name': ensembleName,
                            'method': method,
                            'models': selectedModelNames.copy(),
                            'train_accuracy': trainAcc,
                            'test_accuracy': testAcc,
                            'r2_train': r2Train,
                            'r2_test': r2Test,
                            'problem_type': 'regression'
                        })

                    st.success(f"Ensemble '{ensembleName}' trained! Test Accuracy: {testAcc:.2%}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error training ensemble: {str(e)}")

    # Show trained ensembles
    if st.session_state.ensemble_results:
        st.divider()
        st.markdown("### Trained Ensembles")

        for i, result in enumerate(st.session_state.ensemble_results):
            resultCol1, resultCol2 = st.columns([4, 1])
            with resultCol1:
                st.markdown(f"**{result['name']}** ({result['method']})")
                st.caption(f"Models: {', '.join(result['models'])}")
                col1, col2 = st.columns(2)
                col1.metric("Train Accuracy", f"{result['train_accuracy']:.2%}")
                col2.metric("Test Accuracy", f"{result['test_accuracy']:.2%}")
            with resultCol2:
                if st.button("🗑️", key=f"delete_ensemble_{i}", help="Delete this ensemble"):
                    st.session_state.ensemble_results.pop(i)
                    st.rerun()
            st.divider()

    # Done button
    if st.button("Done", type="primary", use_container_width=True, key="ensemble_done"):
        st.rerun()

# ==================== PAGE 4: MODELS INTERFACE ====================
if st.session_state.page == 'models':
    # Load data (cached)
    @st.cache_data
    def get_data():
        return preprocessingData()

    X_train_base, X_test_base, y_train, y_test, scaler, features = get_data()

    # Convert to DataFrame and add engineered features
    X_train_df = pd.DataFrame(X_train_base, columns=features)
    X_test_df = pd.DataFrame(X_test_base, columns=features)

    # Add engineered features from session state
    engineeredFeatureNames = []
    for engFeature in st.session_state.engineered_features:
        X_train_df[engFeature['name']] = engFeature['values_train']
        X_test_df[engFeature['name']] = engFeature['values_test']
        engineeredFeatureNames.append(engFeature['name'])

    # Update features list
    features = list(X_train_df.columns)

    # Convert back to numpy arrays for model training
    X_train = X_train_df.values
    X_test = X_test_df.values

    # Title with feature engineering, ensemble, visualize, and back buttons
    col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 1])
    with col1:
        if st.session_state.problem_type == 'classification':
            st.title("🚢 Titanic Survival Prediction - Classification Models")
        else:
            st.title("🚢 Titanic Survival Prediction - Regression Models")
    with col2:
        if st.button("✨ Features", help="Open Feature Engineering"):
            featureEngineeringDialog()
    with col3:
        if st.button("🔗 Ensemble", help="Combine multiple models"):
            ensembleLearningDialog()
    with col4:
        if st.button("📊 Visualize", help="Watch training animation"):
            trainingVisualizationDialog()
    with col5:
        if st.button("← Back"):
            st.session_state.page = 'welcome'
            st.rerun()

    if st.session_state.problem_type == 'classification':
        st.markdown("Experiment with different models and hyperparameters to maximize prediction accuracy!")
    else:
        st.markdown("Experiment with different regression models to predict survival probability!")

    # Display dataset statistics
    # Count saved models for current problem type
    savedModelsCount = len([m for m in st.session_state.trained_models.values() if m['problem_type'] == st.session_state.problem_type])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Passengers", len(X_train) + len(X_test))
    col2.metric("Training Samples", len(X_train))
    col3.metric("Test Samples", len(X_test))
    col4.metric("Features", len(features), delta=f"+{len(engineeredFeatureNames)} engineered" if engineeredFeatureNames else None)
    col5.metric("Saved Models", savedModelsCount, help="Models available for ensemble")

    # Show engineered features if any
    if engineeredFeatureNames:
        with st.expander(f"✨ Engineered Features ({len(engineeredFeatureNames)})"):
            for engFeature in st.session_state.engineered_features:
                st.markdown(f"- **{engFeature['name']}**: {engFeature['formula']}")

    # Show ensemble results if any
    if st.session_state.ensemble_results:
        with st.expander(f"🔗 Ensemble Models ({len(st.session_state.ensemble_results)})"):
            for result in st.session_state.ensemble_results:
                if result.get('problem_type') == st.session_state.problem_type:
                    st.markdown(f"**{result['name']}** - {result['method']}")
                    st.caption(f"Models: {', '.join(result['models'])}")
                    metricCol1, metricCol2 = st.columns(2)
                    metricCol1.metric("Train Accuracy", f"{result['train_accuracy']:.2%}")
                    metricCol2.metric("Test Accuracy", f"{result['test_accuracy']:.2%}")
                    st.divider()

    st.divider()

    # Sidebar for model selection (ONLY classification models)
    st.sidebar.title("🎛️ Model Configuration")
    
    if st.session_state.problem_type == 'classification':
        model_choice = st.sidebar.selectbox(
            "Select Classification Model",
            ["Decision Tree", "Random Forest", "Logistic Regression", "SVM", "KNN", "Perceptron", "Neural Network"]
        )
    else:  # Regression
        model_choice = st.sidebar.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression"]
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
                
                # Save model for ensemble
                modelName = f"Decision Tree (depth={maxDepth})"
                st.session_state.trained_models[modelName] = {
                    'model': dtModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'max_depth': maxDepth, 'min_samples_split': minSamplesSplit, 'criterion': criterion},
                    'problem_type': 'classification'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Decision Tree',
                    'model': dtModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'max_depth': maxDepth}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")
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

                # Save model for ensemble
                modelName = f"Random Forest (n={n_estimators})"
                st.session_state.trained_models[modelName] = {
                    'model': rfModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'n_estimators': n_estimators, 'max_depth': maxDepth, 'criterion': criterion},
                    'problem_type': 'classification'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Random Forest',
                    'model': rfModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'n_estimators': n_estimators, 'max_depth': maxDepth}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")
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

                    # Save model for ensemble
                    modelName = f"Logistic Regression (C={C})"
                    st.session_state.trained_models[modelName] = {
                        'model': lrModel,
                        'train_accuracy': trainingAccuracy,
                        'test_accuracy': testingAccuracy,
                        'hyperparams': {'C': C, 'penalty': penalty, 'solver': solver, 'max_iter': max_iter},
                        'problem_type': 'classification'
                    }

                    # Save visualization data
                    st.session_state.last_visualization = {
                        'model_type': 'Logistic Regression',
                        'model': lrModel,
                        'X_train': X_train,
                        'y_train': y_train,
                        'features': features,
                        'hyperparams': {'C': C}
                    }

                    st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

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

                # Save model for ensemble
                modelName = f"SVM (kernel={kernel})"
                st.session_state.trained_models[modelName] = {
                    'model': svmModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'C': C, 'kernel': kernel, 'gamma': gamma},
                    'problem_type': 'classification'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'SVM',
                    'model': svmModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'C': C, 'kernel': kernel}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")


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

                # Save model for ensemble
                modelName = f"KNN (k={n_neighbors})"
                st.session_state.trained_models[modelName] = {
                    'model': knnModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric},
                    'problem_type': 'classification'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'KNN',
                    'model': knnModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'n_neighbors': n_neighbors}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

    ############################################################## Perceptron ########################################################################################
    elif model_choice == "Perceptron":
        st.sidebar.markdown("### Hyperparameters")

        penalty = st.sidebar.selectbox(
            "Penalty",
            [None, "l2", "l1", "elasticnet"],
            help="Regularization term. 'l2' is standard, 'l1' promotes sparsity, 'elasticnet' combines both."
        )

        alpha = st.sidebar.select_slider(
            "Alpha (Regularization Strength)",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.0001,
            help="Constant that multiplies the regularization term. Higher = stronger regularization."
        )

        maxIter = st.sidebar.slider(
            "Max Iterations",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Maximum number of passes over the training data."
        )

        eta0 = st.sidebar.select_slider(
            "Learning Rate (eta0)",
            options=[0.1, 0.5, 1.0, 2.0],
            value=1.0,
            help="Constant by which the updates are multiplied."
        )

        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training Perceptron..."):
                perceptronModel = Perceptron(
                    penalty=penalty,
                    alpha=alpha,
                    max_iter=maxIter,
                    eta0=eta0,
                    random_state=42
                )
                perceptronModel.fit(X_train, y_train)

                yTrainPred = perceptronModel.predict(X_train)
                yTestPred = perceptronModel.predict(X_test)

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
                    help="Difference between training and test accuracy"
                )

                st.markdown("### Interpretation:")
                if accuracyGap > 0.15:
                    st.error("**High overfitting detected!** Try increasing regularization (alpha) or using a penalty.")
                elif accuracyGap > 0.10:
                    st.warning("**Moderate overfitting.** Consider adjusting hyperparameters.")
                elif testingAccuracy < 0.70:
                    st.warning("**Low accuracy.** Perceptron works best on linearly separable data. Try other models.")
                elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                    st.success("**Good job!** Your model generalizes well.")
                else:
                    st.info("Keep experimenting with hyperparameters!")

                st.divider()

                # Confusion Matrix
                st.markdown("### Confusion Matrix (Test Set)")
                with st.expander("Understanding the Confusion Matrix"):
                    st.markdown("""
                    - **Top-left (True Negative):** Correctly predicted deaths
                    - **Top-right (False Positive):** Incorrectly predicted survival
                    - **Bottom-left (False Negative):** Incorrectly predicted death
                    - **Bottom-right (True Positive):** Correctly predicted survival
                    """)

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
                st.divider()

                st.markdown("### Detailed Classification Report")
                report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

                # Save model for ensemble
                modelName = f"Perceptron (alpha={alpha})"
                st.session_state.trained_models[modelName] = {
                    'model': perceptronModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'penalty': penalty, 'alpha': alpha, 'max_iter': maxIter, 'eta0': eta0},
                    'problem_type': 'classification'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Perceptron',
                    'model': perceptronModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'max_iter': maxIter, 'eta0': eta0}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

    ############################################################## Neural Network (MLP) ##############################################################################
    elif model_choice == "Neural Network":
        st.sidebar.markdown("### Hyperparameters")

        numLayers = st.sidebar.slider(
            "Number of Hidden Layers",
            min_value=1,
            max_value=3,
            value=1,
            help="How many hidden layers. More layers = can learn more complex patterns but risk overfitting."
        )

        neuronsPerLayer = st.sidebar.select_slider(
            "Neurons per Layer",
            options=[32, 64, 100, 128, 256],
            value=100,
            help="Number of neurons in each hidden layer. More neurons = more capacity but slower training."
        )

        # Build the tuple dynamically
        hiddenLayerSizes = tuple([neuronsPerLayer] * numLayers)

        activation = st.sidebar.selectbox(
            "Activation Function",
            ["relu", "tanh", "logistic"],
            help="'relu' is most common and fastest. 'tanh' can work better for some problems."
        )

        solver = st.sidebar.selectbox(
            "Solver",
            ["adam", "sgd", "lbfgs"],
            help="'adam' works well for most cases. 'sgd' for more control. 'lbfgs' for smaller datasets."
        )

        alpha = st.sidebar.select_slider(
            "Alpha (L2 Regularization)",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.0001,
            help="L2 penalty parameter. Higher values = stronger regularization."
        )

        maxIter = st.sidebar.slider(
            "Max Iterations",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum number of iterations for training."
        )

        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training Neural Network (this may take a moment)..."):
                mlpModel = MLPClassifier(
                    hidden_layer_sizes=hiddenLayerSizes,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    max_iter=maxIter,
                    random_state=42
                )
                mlpModel.fit(X_train, y_train)

                yTrainPred = mlpModel.predict(X_train)
                yTestPred = mlpModel.predict(X_test)

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
                    help="Difference between training and test accuracy"
                )

                st.markdown("### Interpretation:")
                if accuracyGap > 0.15:
                    st.error("**High overfitting!** Try increasing alpha, reducing hidden layer size, or using fewer iterations.")
                elif accuracyGap > 0.10:
                    st.warning("**Moderate overfitting.** Consider simpler architecture or more regularization.")
                elif testingAccuracy < 0.70:
                    st.warning("**Low accuracy.** Try a different architecture or more iterations.")
                elif accuracyGap < 0.05 and testingAccuracy > 0.75:
                    st.success("**Excellent!** Your neural network generalizes well.")
                else:
                    st.info("Keep experimenting with different architectures!")

                st.divider()

                # Confusion Matrix
                st.markdown("### Confusion Matrix (Test Set)")
                with st.expander("Understanding the Confusion Matrix"):
                    st.markdown("""
                    - **Top-left (True Negative):** Correctly predicted deaths
                    - **Top-right (False Positive):** Incorrectly predicted survival
                    - **Bottom-left (False Negative):** Incorrectly predicted death
                    - **Bottom-right (True Positive):** Correctly predicted survival
                    """)

                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='RdPu',
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

                st.markdown("### Detailed Classification Report")
                report = classification_report(y_test, yTestPred, target_names=['Died', 'Survived'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

                # Save model for ensemble
                layerDisplay = ", ".join([str(neuronsPerLayer)] * numLayers)
                modelName = f"Neural Network ({layerDisplay})"
                st.session_state.trained_models[modelName] = {
                    'model': mlpModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'hidden_layer_sizes': hiddenLayerSizes, 'num_layers': numLayers, 'neurons_per_layer': neuronsPerLayer, 'activation': activation, 'solver': solver, 'alpha': alpha, 'max_iter': maxIter},
                    'problem_type': 'classification'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Neural Network',
                    'model': mlpModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'hidden_layer_sizes': hiddenLayerSizes, 'activation': activation, 'max_iter': maxIter}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

    ############################################################## Linear Regression #################################################################################
    elif model_choice == "Linear Regression":
        st.sidebar.markdown("### Hyperparameters")
        st.sidebar.info("Linear Regression has no hyperparameters to tune. It finds the best-fit line automatically.")

        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training Linear Regression..."):
                linearModel = LinearRegression()
                linearModel.fit(X_train, y_train)

                yTrainPred = linearModel.predict(X_train)
                yTestPred = linearModel.predict(X_test)

                # Convert to binary for accuracy comparison
                yTrainPredBinary = (yTrainPred >= 0.5).astype(int)
                yTestPredBinary = (yTestPred >= 0.5).astype(int)

                trainingAccuracy = accuracy_score(y_train, yTrainPredBinary)
                testingAccuracy = accuracy_score(y_test, yTestPredBinary)

                # Regression metrics
                r2Train = r2_score(y_train, yTrainPred)
                r2Test = r2_score(y_test, yTestPred)
                mseTest = mean_squared_error(y_test, yTestPred)
                maeTest = mean_absolute_error(y_test, yTestPred)

                st.success("Model trained successfully!")

                st.markdown("### Regression Metrics:")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("R² (Train)", f"{r2Train:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**R² (Train)**")
                        st.write("R-squared measures how well the model explains variance in the training data. 1.0 = perfect fit, 0.0 = no explanatory power. Higher is better, but very high values may indicate overfitting.")
                with c2:
                    st.metric("R² (Test)", f"{r2Test:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**R² (Test)**")
                        st.write("R-squared on unseen test data. Shows how well the model generalizes. Compare with R² (Train) - a large gap suggests overfitting.")
                with c3:
                    st.metric("MSE", f"{mseTest:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**Mean Squared Error**")
                        st.write("Average of squared differences between predicted and actual values. Penalizes larger errors more heavily. Lower is better.")
                with c4:
                    st.metric("MAE", f"{maeTest:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**Mean Absolute Error**")
                        st.write("Average of absolute differences between predicted and actual values. More interpretable than MSE. Lower is better.")

                st.divider()

                st.markdown("### Classification Accuracy (threshold = 0.5):")
                c1, c2 = st.columns(2)
                c1.metric("Training Accuracy", f"{trainingAccuracy:.2%}")
                c2.metric("Test Accuracy", f"{testingAccuracy:.2%}")

                st.markdown("### Interpretation:")
                if r2Test > 0.3:
                    st.success("**Good fit!** The model explains a reasonable amount of variance.")
                else:
                    st.warning("**Low R².** Linear regression may not be ideal for this binary classification problem.")

                st.divider()

                # Confusion Matrix
                st.markdown("### Confusion Matrix (Test Set)")
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPredBinary)
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

                # Save model for ensemble
                modelName = "Linear Regression"
                st.session_state.trained_models[modelName] = {
                    'model': linearModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {},
                    'problem_type': 'regression'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Linear Regression',
                    'model': linearModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

    ############################################################## Ridge Regression ##################################################################################
    elif model_choice == "Ridge Regression":
        st.sidebar.markdown("### Hyperparameters")

        alpha = st.sidebar.select_slider(
            "Alpha (Regularization Strength)",
            options=[0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            help="L2 regularization strength. Higher = more regularization, simpler model."
        )

        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training Ridge Regression..."):
                ridgeModel = Ridge(alpha=alpha, random_state=42)
                ridgeModel.fit(X_train, y_train)

                yTrainPred = ridgeModel.predict(X_train)
                yTestPred = ridgeModel.predict(X_test)

                yTrainPredBinary = (yTrainPred >= 0.5).astype(int)
                yTestPredBinary = (yTestPred >= 0.5).astype(int)

                trainingAccuracy = accuracy_score(y_train, yTrainPredBinary)
                testingAccuracy = accuracy_score(y_test, yTestPredBinary)

                r2Train = r2_score(y_train, yTrainPred)
                r2Test = r2_score(y_test, yTestPred)
                mseTest = mean_squared_error(y_test, yTestPred)
                maeTest = mean_absolute_error(y_test, yTestPred)

                st.success("Model trained successfully!")

                st.markdown("### Regression Metrics:")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("R² (Train)", f"{r2Train:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**R² (Train)**")
                        st.write("R-squared measures how well the model explains variance in the training data. 1.0 = perfect fit, 0.0 = no explanatory power. Higher is better, but very high values may indicate overfitting.")
                with c2:
                    st.metric("R² (Test)", f"{r2Test:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**R² (Test)**")
                        st.write("R-squared on unseen test data. Shows how well the model generalizes. Compare with R² (Train) - a large gap suggests overfitting.")
                with c3:
                    st.metric("MSE", f"{mseTest:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**Mean Squared Error**")
                        st.write("Average of squared differences between predicted and actual values. Penalizes larger errors more heavily. Lower is better.")
                with c4:
                    st.metric("MAE", f"{maeTest:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**Mean Absolute Error**")
                        st.write("Average of absolute differences between predicted and actual values. More interpretable than MSE. Lower is better.")

                st.divider()

                st.markdown("### Classification Accuracy (threshold = 0.5):")
                c1, c2 = st.columns(2)
                c1.metric("Training Accuracy", f"{trainingAccuracy:.2%}")
                c2.metric("Test Accuracy", f"{testingAccuracy:.2%}")

                st.markdown("### Interpretation:")
                st.info(f"**Ridge Regression** uses L2 regularization to prevent overfitting. Alpha={alpha} controls the regularization strength.")

                st.divider()

                st.markdown("### Confusion Matrix (Test Set)")
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPredBinary)
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

                # Save model for ensemble
                modelName = f"Ridge Regression (alpha={alpha})"
                st.session_state.trained_models[modelName] = {
                    'model': ridgeModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'alpha': alpha},
                    'problem_type': 'regression'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Ridge Regression',
                    'model': ridgeModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'alpha': alpha}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

    ############################################################## Lasso Regression ##################################################################################
    elif model_choice == "Lasso Regression":
        st.sidebar.markdown("### Hyperparameters")

        alpha = st.sidebar.select_slider(
            "Alpha (Regularization Strength)",
            options=[0.0001, 0.001, 0.01, 0.1, 1.0],
            value=0.01,
            help="L1 regularization strength. Higher = more features set to zero (feature selection)."
        )

        maxIter = st.sidebar.slider(
            "Max Iterations",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000,
            help="Maximum number of iterations for convergence."
        )

        if st.sidebar.button("Train Model", type="primary"):
            with st.spinner("Training Lasso Regression..."):
                lassoModel = Lasso(alpha=alpha, max_iter=maxIter, random_state=42)
                lassoModel.fit(X_train, y_train)

                yTrainPred = lassoModel.predict(X_train)
                yTestPred = lassoModel.predict(X_test)

                yTrainPredBinary = (yTrainPred >= 0.5).astype(int)
                yTestPredBinary = (yTestPred >= 0.5).astype(int)

                trainingAccuracy = accuracy_score(y_train, yTrainPredBinary)
                testingAccuracy = accuracy_score(y_test, yTestPredBinary)

                r2Train = r2_score(y_train, yTrainPred)
                r2Test = r2_score(y_test, yTestPred)
                mseTest = mean_squared_error(y_test, yTestPred)
                maeTest = mean_absolute_error(y_test, yTestPred)

                st.success("Model trained successfully!")

                st.markdown("### Regression Metrics:")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("R² (Train)", f"{r2Train:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**R² (Train)**")
                        st.write("R-squared measures how well the model explains variance in the training data. 1.0 = perfect fit, 0.0 = no explanatory power. Higher is better, but very high values may indicate overfitting.")
                with c2:
                    st.metric("R² (Test)", f"{r2Test:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**R² (Test)**")
                        st.write("R-squared on unseen test data. Shows how well the model generalizes. Compare with R² (Train) - a large gap suggests overfitting.")
                with c3:
                    st.metric("MSE", f"{mseTest:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**Mean Squared Error**")
                        st.write("Average of squared differences between predicted and actual values. Penalizes larger errors more heavily. Lower is better.")
                with c4:
                    st.metric("MAE", f"{maeTest:.4f}")
                    with st.popover("ℹ️"):
                        st.markdown("**Mean Absolute Error**")
                        st.write("Average of absolute differences between predicted and actual values. More interpretable than MSE. Lower is better.")

                st.divider()

                st.markdown("### Classification Accuracy (threshold = 0.5):")
                c1, c2 = st.columns(2)
                c1.metric("Training Accuracy", f"{trainingAccuracy:.2%}")
                c2.metric("Test Accuracy", f"{testingAccuracy:.2%}")

                st.markdown("### Interpretation:")
                st.info(f"**Lasso Regression** uses L1 regularization which can set some feature coefficients to exactly zero, performing automatic feature selection.")

                # Show feature coefficients
                st.markdown("### Feature Coefficients:")
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': lassoModel.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                st.dataframe(coef_df, use_container_width=True)

                st.divider()

                st.markdown("### Confusion Matrix (Test Set)")
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, yTestPredBinary)
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

                # Save model for ensemble
                modelName = f"Lasso Regression (alpha={alpha})"
                st.session_state.trained_models[modelName] = {
                    'model': lassoModel,
                    'train_accuracy': trainingAccuracy,
                    'test_accuracy': testingAccuracy,
                    'hyperparams': {'alpha': alpha, 'max_iter': maxIter},
                    'problem_type': 'regression'
                }

                # Save visualization data
                st.session_state.last_visualization = {
                    'model_type': 'Lasso Regression',
                    'model': lassoModel,
                    'X_train': X_train,
                    'y_train': y_train,
                    'features': features,
                    'hyperparams': {'alpha': alpha}
                }

                st.success(f"Model trained and saved! Ready for ensemble. Click 📊 Visualize button to see training animation.")

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