import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------

with open("model_pickle.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ LOAD DATA ------------------

df = pd.read_csv("laptop_data_cleaned.csv")

# ------------------ CUSTOM CSS ------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:#1565C0;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.stButton>button{
    width:100%;
    background-color:#1565C0;
    color:white;
    font-size:20px;
    border-radius:10px;
    height:3.2em;
}

.stButton>button:hover{
    background-color:#0D47A1;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------

st.markdown("<div class='title'>💻 Laptop Price Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Random Forest Regression Model</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ SIDEBAR ------------------

st.sidebar.header("Project Information")

st.sidebar.success("Model : Random Forest Regressor")

st.sidebar.write("Dataset Size : 1273")

st.sidebar.write("Features : 12")

st.sidebar.write("Target : Laptop Price")

st.sidebar.markdown("---")

st.sidebar.write("Developed using Streamlit")

# ------------------ INPUTS ------------------

col1, col2 = st.columns(2)

with col1:

    company = st.selectbox(
        "Company",
        sorted(df["Company"].unique())
    )

    typename = st.selectbox(
        "Laptop Type",
        sorted(df["TypeName"].unique())
    )

    ram = st.selectbox(
        "RAM (GB)",
        sorted(df["Ram"].unique())
    )

    weight = st.slider(
        "Weight (kg)",
        float(df["Weight"].min()),
        float(df["Weight"].max()),
        float(df["Weight"].median()),
        0.01
    )

    touchscreen = st.selectbox(
        "Touch Screen",
        [0,1],
        format_func=lambda x: "Yes" if x==1 else "No"
    )

    ips = st.selectbox(
        "IPS Display",
        [0,1],
        format_func=lambda x: "Yes" if x==1 else "No"
    )

with col2:

    ppi = st.slider(
        "PPI",
        float(df["Ppi"].min()),
        float(df["Ppi"].max()),
        float(df["Ppi"].median()),
        0.1
    )

    cpu = st.selectbox(
        "CPU Brand",
        sorted(df["Cpu_brand"].unique())
    )

    hdd = st.selectbox(
        "HDD (GB)",
        sorted(df["HDD"].unique())
    )

    ssd = st.selectbox(
        "SSD (GB)",
        sorted(df["SSD"].unique())
    )

    gpu = st.selectbox(
        "GPU Brand",
        sorted(df["Gpu_brand"].unique())
    )

    os = st.selectbox(
        "Operating System",
        sorted(df["Os"].unique())
    )

st.markdown("---")

# ------------------ PREDICT ------------------

if st.button("🔍 Predict Laptop Price"):

    input_df = pd.DataFrame({

        "Company":[company],
        "TypeName":[typename],
        "Ram":[ram],
        "Weight":[weight],
        "TouchScreen":[touchscreen],
        "Ips":[ips],
        "Ppi":[ppi],
        "Cpu_brand":[cpu],
        "HDD":[hdd],
        "SSD":[ssd],
        "Gpu_brand":[gpu],
        "Os":[os]

    })

    prediction = model.predict(input_df)[0]

    actual_price = np.exp(prediction)

    st.success("Prediction Successful ✅")

    st.markdown("## 💰 Estimated Laptop Price")

    st.metric(
        label="Predicted Price",
        value=f"₹ {actual_price:,.2f}"
    )

    st.balloons()

# ------------------ FOOTER ------------------

st.markdown("---")

st.markdown(
"""
<center>

### 💻 Laptop Price Prediction using Random Forest

Developed with ❤️ using Streamlit

</center>
""",
unsafe_allow_html=True
)