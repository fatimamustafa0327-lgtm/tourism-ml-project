import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tourism Insights", page_icon="🌍")

st.title("🌍 Tourism Insights & Prediction")
st.write("Explore tourism data and view Decision Tree predictions.")

@st.cache_data
def load_data():
    return pd.read_csv("tourism_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.pkl")

try:
    df = load_data()
    st.write("Columns:",
             df.columns.tolist())
    model = load_model()

    st.subheader("📊 Tourism Dataset")
    st.write("Total records:", len(df))
    st.dataframe(df.head(10))

    st.subheader("🥧 Tourism Pie Chart")
    columns = df.select_dtypes(include="object").columns.tolist()

    if columns:
        chart_col = st.selectbox("Choose a category", columns)
        counts = df[chart_col].fillna("Unknown").value_counts().head(10)

        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        ax.set_title(f"Distribution of {chart_col}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No text category columns were found for the pie chart.")

    st.subheader("🤖 Decision Tree Model")
    st.write("Model loaded successfully!")
    st.subheader("🔮 Tourism Prediction")

st.write("Enter tourism details below:")

visitors = st.number_input("Visitors", min_value=0, value=1000)
rating = st.slider("Rating", 0.0, 5.0, 3.0)
revenue = st.number_input("Revenue", min_value=0.0, value=1000.0)

location = st.text_input("Location")
country = st.text_input("Country")
category = st.text_input("Category")
accommodation = st.selectbox(
    "Accommodation Available", ["Yes", "No"]
)

if st.button("Predict"):
    st.warning(
        "Prediction form is ready, but the model's exact training "
        "columns must be matched before predictions can run."
    )

    st.info(
        "Your saved model is ready to connect to the prediction form. "
        "We need to confirm the original training features and preprocessing "
        "before making predictions."
    )

except Exception as e:
    st.error("Could not load the dataset or model.")
    st.write(str(e))
