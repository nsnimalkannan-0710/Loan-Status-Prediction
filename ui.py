import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, silhouette_score,
                             mean_squared_error, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os


# App Configuration
st.set_page_config(page_title="Loan Prediction System", layout="wide")
st.title("Loan Prediction System")

# Tab titles
tab1, tab2, tab3, tab4 = st.tabs(["Loan Approval", "Customer Segmentation", "Amount Calculator","Featurn Distribution"])
def set_custom_style():
    st.markdown("""
    <style>
    /* Tabs Container */
    [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0f2f6;
        padding: 8px !important;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    /* Individual Tabs */
    [data-baseweb="tab"] {
        padding: 10px 20px !important;
        background: white !important;
        border-radius: 8px !important;
        margin: 0 !important;
        border: 1px solid #d0d0d0 !important;
        font-weight: 500;
        color: #444 !important;
        transition: all 0.3s;
    }
    
    /* Hover State */
    [data-baseweb="tab"]:hover {
        background: #f8f9fa !important;
        color: #222 !important;
        border-color: #b0b0b0 !important;
    }
    
    /* Active Tab */
    [aria-selected="true"] {
        background: #6a11cb !important;
        color: white !important;
        border-color: #6a11cb !important;
        box-shadow: 0 2px 8px rgba(106,17,203,0.2);
    }
    
    /* Tab Icons */
    [data-baseweb="tab"] svg {
        margin-right: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_custom_style()  # Call this right after set_page_config()


# Data Loading
@st.cache_data
def load_data():
    # Read CSV from specified path
    try:
        df = pd.read_csv('data/loan_data.csv')  # Update with your actual path
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'loan_data.csv' exists in the specified path.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Data Preprocessing
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Convert categorical columns
    label_encoder = LabelEncoder()
    if 'education' in df.columns:
        df['education'] = label_encoder.fit_transform(df['education'])
    if 'self_employed' in df.columns:
        df['self_employed'] = label_encoder.fit_transform(df['self_employed'])
    if 'loan_status' in df.columns:
        df['loan_status'] = label_encoder.fit_transform(df['loan_status'])

    # Train models
    X_class = df[[   'income_annum', 'loan_amount',
                 'loan_term', 'cibil_score', 'residential_assets_value' ]]
    y_class = df['loan_status']
    
    X_cluster = df[['income_annum', 'loan_amount', 'cibil_score', 
                   'residential_assets_value', 'commercial_assets_value']]
    
    X_reg = df[['no_of_dependents', 'education', 'self_employed', 'income_annum', 
               'loan_term', 'cibil_score', 'residential_assets_value']]
    y_reg = df['loan_amount']

    # Classification Model (Loan Approval)
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42)
    clf_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf_model.fit(X_train_class, y_train_class)
    y_pred_class = clf_model.predict(X_test_class)
    clf_accuracy = accuracy_score(y_test_class, y_pred_class)

    # Clustering Model (Customer Segmentation)
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
    clusters = cluster_model.fit_predict(X_cluster_scaled)
    silhouette = silhouette_score(X_cluster_scaled, clusters)

    # Regression Model (Amount Calculator)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train_reg)
    X_test_poly = poly_features.transform(X_test_reg)
    reg_model = LinearRegression()
    reg_model.fit(X_train_poly, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_poly)
    reg_mse = mean_squared_error(y_test_reg, y_pred_reg)
    reg_r2 = r2_score(y_test_reg, y_pred_reg)

    # Loan Approval Tab
    with tab1:
        st.header("Loan Approval Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
           # no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
           # education = st.selectbox("Education", ["Graduate", "Not Graduate"])
           # self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Annual Income (USD)", min_value=0, value=50000)
            loan_amount = st.number_input("Loan Amount (USD)", min_value=0, value=50000000)
           
            loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=5)
            
        with col2:
           
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
            residential_assets = st.number_input("Residential Assets Value (USD)", min_value=0, value=20000)
            #commercial_assets = st.number_input("Commercial Assets Value (USD)", min_value=0, value=50000)
            
        
        if st.button("Predict Loan Approval"):
            # Prepare input
            #education_encoded = 1 if education == "Graduate" else 0
            #self_employed_encoded = 1 if self_employed == "Yes" else 0
            
            # Remove loan_amount from the input data (as it was not part of the training data)
            input_data = [[ 
                          income_annum,loan_amount, loan_term, cibil_score, residential_assets
                          ]]  # Exclude loan_amount
            
            # Make prediction
            prediction = clf_model.predict(input_data)
            probability = clf_model.predict_proba(input_data)
            
            # Display results
            if prediction[0] == 1:
                st.success("Loan Status: Approved")
                st.write(f"Confidence: {probability[0][1]:.2%}")
                st.write("Suggestion: Your application looks strong. Consider proceeding with the loan process.")
            else:
                st.error("Loan Status: Rejected")
                st.write(f"Confidence: {probability[0][0]:.2%}")
                st.write("Suggestions:")
                st.write("- Improve your CIBIL score (aim for 750+)")
                st.write("- Reduce the loan amount or increase loan term")
                st.write("- Increase your asset values to improve collateral")

    # Customer Segmentation Tab
    with tab2:
        st.header("Customer Segmentation")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            seg_income = st.number_input("Annual Income (USD)", key='seg_income', min_value=0, value=50000)
            seg_loan_amount = st.number_input("Loan Amount (USD)", key='seg_loan', min_value=0, value=100000)
            
        with col2:
            seg_cibil = st.number_input("CIBIL Score", key='seg_cibil', min_value=300, max_value=900, value=700)
            seg_res_assets = st.number_input("Residential Assets (USD)", key='seg_res', min_value=0, value=20000)
            seg_com_assets = st.number_input("Commercial Assets (USD)", key='seg_com', min_value=0, value=50000)
        
        if st.button("Segment Customer"):
            # Prepare input by combining with existing data
            input_data = [[seg_income, seg_loan_amount, seg_cibil, seg_res_assets, seg_com_assets]]
            combined_data = np.vstack([X_cluster_scaled, scaler.transform(input_data)])
            
            # Refit model on combined data
            cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
            clusters = cluster_model.fit_predict(combined_data)
            
            # Get prediction for our specific customer (last one in array)
            cluster = clusters[-1]
            
            # Display results
            if cluster == 0:
                st.warning("Customer Segment: High Risk")
                st.write("Suggestions:")
                st.write("- Consider offering smaller loan amounts")
                st.write("- Require additional collateral")
                st.write("- Higher interest rate recommended")
            else:
                st.success("Customer Segment: Low Risk")
                st.write("Suggestions:")
                st.write("- Can offer competitive interest rates")
                st.write("- Larger loan amounts can be considered")
                st.write("- Faster approval process recommended")

    # Amount Calculator Tab
    with tab3:
        st.header("Loan Amount Calculator : " )
        
        col1, col2 = st.columns(2)
        
        with col1:
            amt_dependents = st.number_input("Number of Dependents", key='amt_dep', min_value=0, max_value=10, value=1)
            amt_education = st.selectbox("Education", ["Graduate", "Not Graduate"], key='amt_edu')
            amt_self_employed = st.selectbox("Self Employed", ["Yes", "No"], key='amt_emp')
            
        with col2:
            amt_income = st.number_input("Annual Income (USD)", key='amt_inc', min_value=0, value=50000)
            amt_term = st.number_input("Loan Term (years)", key='amt_term', min_value=1, max_value=30, value=5)
            amt_cibil = st.number_input("CIBIL Score", key='amt_cibil', min_value=300, max_value=900, value=700)
            amt_res_assets = st.number_input("Residential Assets (USD)", key='amt_res', min_value=0, value=20000)
        
        if st.button("Calculate Loan Amount"):
            # Prepare input
            amt_education_encoded = 1 if amt_education == "Graduate" else 0
            amt_self_employed_encoded = 1 if amt_self_employed == "Yes" else 0
            
            input_data = [[amt_dependents, amt_education_encoded, amt_self_employed_encoded,
                         amt_income, amt_term, amt_cibil, amt_res_assets]]
            input_poly = poly_features.transform(input_data)
            
            # Make prediction
            amount = reg_model.predict(input_poly)
            amount[0]/=100
            # Display results
            st.success(f"Recommended Loan Amount: ${amount[0]:,.2f}")
            st.write("Suggestions:")
            st.write("- This amount is based on your financial profile")
            st.write("- You may qualify for up to 20% more with additional collateral")
            st.write("- Consider a co-signer if you need a larger amount")

# ... (keep your existing code for tabs 1-3) ...

    # Feature Distribution Tab
    with tab4:
        st.header("Feature Distributions")
        st.write("Explore how key features are distributed across applicants.")

        # Select feature to plot
        feature = st.selectbox(
            "Select a feature to visualize:",
            options=["income_annum", "loan_amount", "cibil_score", "residential_assets_value"],
            format_func=lambda x: x.replace("_", " ").title()  # Format display (e.g., "income_annum" -> "Income Annum")
        )

        # Plot settings
        plot_type = st.radio("Plot type:", ["Histogram", "KDE Plot"])
        color = st.color_picker("Pick a color", "#4f8bf9")  # Default: blue

        # Generate the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if plot_type == "Histogram":
            sns.histplot(df[feature], kde=False, color=color, bins=20, ax=ax)
            ax.set_title(f"Distribution of {feature.replace('_', ' ').title()}", fontsize=14)
        else:
            sns.kdeplot(df[feature], color=color, fill=True, ax=ax)
            ax.set_title(f"Density of {feature.replace('_', ' ').title()}", fontsize=14)

        ax.set_xlabel(feature.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.3)
        
        st.pyplot(fig)

        # Add insights based on the feature
        st.subheader("Insights")
        if feature == "income_annum":
            st.write("- Most applicants have incomes between ${:,.0f} and ${:,.0f}.".format(
                df["income_annum"].quantile(0.25), 
                df["income_annum"].quantile(0.75)
            ))
        elif feature == "loan_amount":
            st.write("- Loan amounts are right-skewed, indicating fewer applicants request very large loans.")
        elif feature == "cibil_score":
            st.write("- Scores below 600 are rare (likely filtered during application).")
else:
    st.warning("Please ensure the dataset is available to proceed.")
