import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
def load_model():
    try:
        model = joblib.load('best_diabetes_model.sav')
        return model
    except FileNotFoundError:
        st.error("Model file 'best_diabetes_model.sav' not found. Please ensure the file is in the same directory.")
        st.stop()

model = load_model()

# Main layout
st.title("Diabetes Risk Prediction System")
st.markdown("_Machine Learning Model for Healthcare Risk Assessment_")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Information", "Dataset Information"])

# Tab 1: Prediction
with tab1:
    st.header("Patient Health Assessment")
    
    # Single column input form
    with st.form("patient_form"):
        st.subheader("Demographic Information")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=40, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.subheader("Body Metrics")
        bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
        st.subheader("Blood Pressure")
        blood_pressure_category = st.selectbox(
            "Blood Pressure Category",
            ["Normal", "Hypertension Stage 1", "Hypertension Stage 2", "Elevated"]
        )
        
        st.subheader("Cholesterol")
        cholesterol = st.slider("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
        
        st.subheader("Blood Glucose")
        blood_glucose = st.slider("Blood Glucose Level (mg/dL)", min_value=70, max_value=400, value=120, step=1)
        
        predict_button = st.form_submit_button("Predict Diabetes Risk", use_container_width=True)
    
    if predict_button:
        try:
            # Prepare input data
            input_data = {
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'blood_pressure_category': blood_pressure_category,
                'cholesterol': cholesterol,
                'glucose': blood_glucose
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            diabetes_probability = model.predict(input_df)[0]
            diabetes_probability = np.clip(diabetes_probability, 0, 1)
            
            st.markdown("---")
            st.header("Prediction Results")
            
            # Risk classification
            if diabetes_probability < 0.3:
                risk_level = "Low Risk"
                risk_class = "risk-low"
                recommendation = "Continue maintaining healthy lifestyle habits. Consider regular health check-ups every 2-3 years."
            elif diabetes_probability < 0.7:
                risk_level = "Medium Risk"
                risk_class = "risk-medium"
                recommendation = "Increase physical activity and monitor diet closely. Annual health check-ups recommended."
            else:
                risk_level = "High Risk"
                risk_class = "risk-high"
                recommendation = "Consult healthcare provider immediately for comprehensive evaluation and lifestyle intervention."
            
            # Results metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">Diabetes Probability</h3>
                    <h1 style="margin: 0; color: #007bff;">{diabetes_probability:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h3 style="margin: 0;">Risk Level</h3>
                    <h2 style="margin: 0; color: #333;">{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                health_score = (1 - diabetes_probability) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">Health Score</h3>
                    <h1 style="margin: 0; color: #28a745;">{health_score:.0f}/100</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("Health Insights")
            st.markdown(f"**Recommendation:** {recommendation}")
            
            # Risk factors analysis
            risk_factors = {
                'Glucose Level': min(blood_glucose / 400, 1.0),
                'BMI': min(bmi / 40, 1.0),
                'Cholesterol': min(cholesterol / 400, 1.0),
                'Age': min(age / 100, 1.0)
            }
            
            st.subheader("Contributing Risk Factors")
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Normalized Risk'])
            st.bar_chart(risk_df.set_index('Factor'))
            
            # Specific insights
            st.subheader("Specific Health Insights")
            insights = []
            
            if blood_glucose > 150:
                insights.append("High blood glucose level detected. Monitor carbohydrate intake.")
            if bmi > 25:
                insights.append(f"BMI {bmi:.1f} indicates overweight status. Consider exercise and diet modifications.")
            if cholesterol > 200:
                insights.append(f"High cholesterol ({cholesterol} mg/dL). Increase dietary fiber intake.")
            if age > 45:
                insights.append(f"Age {age} years. Diabetes risk increases with age.")
            
            if not insights:
                st.success("All health indicators are within normal ranges. Excellent!")
            else:
                for insight in insights:
                    st.warning(insight)
            
            st.markdown("---")
            st.info(
                "Disclaimer: This prediction tool is for educational and informational purposes only. "
                "It should not replace professional medical advice, diagnosis, or treatment. "
                "Always consult with qualified healthcare providers for medical decisions."
            )
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please check input values and model file.")

with tab2:
    st.header("Model Performance and Technical Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "Gradient Boosting Regressor")
    
    with col2:
        st.metric("Test R² Score", "0.99")
    
    with col3:
        st.metric("Test MAPE", "0.0012")
    
    st.markdown("---")
    
    st.subheader("Features Used")
    st.write("""
    **Numerical Features:**
    - age (18-100 years)
    - bmi (10.0-60.0)
    - systolic (80-200 mmHg)
    - diastolic (50-130 mmHg)
    - cholesterol (100-400 mg/dL)
    - glucose (70-400 mg/dL)
    
    **Categorical Features:**
    - gender: Male, Female
    - bmi_category: Underweight, Normal, Overweight, Obesity
    - blood_pressure_category: Normal, Hypertension stage I and II, Elevated
    - cholesterol_category: Normal, High, Very High
    """)
    
    st.subheader("Model Metrics Explained")
    st.write("""
    **R² Score (0.99):** Model explains 99% of variance in diabetes probability.
    **MAPE (0.0012):** Average prediction error of 0.12%.
    **MAE:** Average absolute error of 0.0005 probability units.
    
    The model demonstrates excellent predictive performance with minimal overfitting.
    """)

# Tab 3: Dataset Information
with tab3:
    st.header("Dataset Overview")
    
    st.subheader("Source Information")
    st.write("""
    **Dataset:** Comprehensive Dataset for Diabetes Prediction
    **Source:** Kaggle (https://www.kaggle.com/datasets/str1k3rfl0/comprehensive-dataset-for-diabetes-prediction)
    **Size:** 10,000 simulated patient observations
    **Features:** 10 input variables + 1 target (diabetes_probability)
    **Target:** Continuous probability (0-1)
    """)
    
    st.subheader("Input Ranges")
    st.write("""
    - Age: 18-100 years
    - BMI: 10.0-60.0 kg/m²
    - Systolic BP: 80-200 mmHg
    - Diastolic BP: 50-130 mmHg
    - Cholesterol: 100-400 mg/dL
    - Glucose: 70-400 mg/dL
    """)
    
    st.subheader("Risk Classification")
    st.write("""
    Low Risk: < 30% probability
    Medium Risk: 30-70% probability  
    High Risk: > 70% probability
    """)

st.markdown("---")
st.markdown(
    "Diabetes Risk Prediction System | Machine Learning Portfolio Project | "
    "For educational purposes only - consult healthcare professionals for medical advice."
)
