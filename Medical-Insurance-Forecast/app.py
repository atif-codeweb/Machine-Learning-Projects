
import streamlit as st
import joblib
import pandas as pd

# Load models
try:
    poly_transformer = joblib.load('poly_transformer.pkl')
    linear_model = joblib.load('linear_model.pkl')
    poly_model = joblib.load('poly_model.pkl')
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()


st.set_page_config(page_title="Insurance Cost Predictor", layout="wide")
st.title("🏥 Medical Insurance Cost Prediction")


SEX_MAP = {0: "Male", 1: "Female"}
SMOKER_MAP = {0: "Non-smoker", 1: "Smoker"}
REGION_MAP = {
    0: "Southwest", 
    1: "Southeast", 
    2: "Northwest", 
    3: "Northeast"
}


with st.sidebar:
    st.header("Patient Details")
    age = st.slider("Age", 18, 70, 30)
    sex = st.radio("Sex", options=list(SEX_MAP.keys()), format_func=lambda x: SEX_MAP[x])
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)
    children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
    smoker = st.radio("Smoker Status", options=list(SMOKER_MAP.keys()), format_func=lambda x: SMOKER_MAP[x])
    region = st.selectbox("Region", options=list(REGION_MAP.keys()), format_func=lambda x: REGION_MAP[x])
    model_type = st.radio("Model Type", ["Linear Regression", "Polynomial Regression"])


input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})


input_df = input_df.astype({
    'age': 'float64',
    'bmi': 'float64',
    'children': 'int64',
    'sex': 'int64',
    'smoker': 'int64',
    'region': 'int64'
})


if st.sidebar.button("Predict Cost"):
    try:
        if model_type == "Linear Regression":
            prediction = linear_model.predict(input_df)[0]
        else:
            x_poly = poly_transformer.transform(input_df)
            prediction = poly_model.predict(x_poly)[0]

        st.success(f"Predicted Insurance Cost: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Debug Info:")
        st.write("Input data used:", input_df)
        st.write("Data types:", input_df.dtypes)


with st.expander("Model Information"):
    st.write("""
    **Feature Encoding:**
    - Sex: 0 = Male, 1 = Female
    - Smoker: 0 = Non-smoker, 1 = Smoker
    - Region: 
        - 0 = Southwest
        - 1 = Southeast
        - 2 = Northwest
        - 3 = Northeast
    """)

    st.write("""
    **Model Performance:**
    - Linear Regression R²: 0.78
    - Polynomial Regression R²: 0.85
    """)


st.markdown("""
<style>
    .stSuccess { padding: 0.5rem; border-radius: 0.5rem; }
    .stAlert { padding: 0.5rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

