from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "C:/Machine learning/Classification/berlin_housing_data.csv"
SUBMISSIONS_PATH = BASE_DIR / "customer_inquiries.csv"
FEATURE_COLUMNS = ["Square_Feet", "Bedrooms", "Neighborhood"]

st.set_page_config(page_title="Home Estimate | Property Desk",
                   page_icon="⌂", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    :root {
        --ink: #172c27;
        --muted: #63736e;
        --green: #176b52;
        --green-dark: #104c3d;
        --coral: #d96f4b;
        --line: #dce5df;
        --paper: #f5f8f5;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 2.2rem; }
    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: 'DM Serif Display', Georgia, serif; font-size: 2.8rem; font-weight: 400; }
    h2, h3, label, p, input, button { font-family: 'DM Sans', sans-serif; }
    .eyebrow {
        color: var(--green); font-size: .75rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .45rem;
    }
    .intro { color: var(--muted); margin-top: -.55rem; }
    .section-title {
        border-bottom: 1px solid var(--line); color: var(--green-dark);
        font-size: 1.02rem; font-weight: 700; margin: .7rem 0 1rem; padding-bottom: .65rem;
    }
    div[data-testid="stForm"] {
        background: #ffffff; border: 1px solid var(--line); border-radius: 8px;
        padding: 1.4rem 1.6rem 1.1rem;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: var(--line); border-radius: 5px;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: var(--green); border: 1px solid var(--green);
        border-radius: 5px; color: white; font-weight: 700; min-height: 2.8rem;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--green-dark); border-color: var(--green-dark); color: white;
    }
    .estimate-label { color: var(--muted); font-size: .85rem; margin-bottom: .2rem; }
    .estimate-value {
        color: var(--green-dark); font-family: 'DM Serif Display', Georgia, serif;
        font-size: 2.6rem; line-height: 1.2;
    }
    .estimate-rule { border-left: 3px solid var(--coral); padding: .2rem 0 .25rem 1rem; }
    @media (max-width: 700px) {
        h1 { font-size: 2.2rem; }
        div[data-testid="stForm"] { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> tuple[object, list[str]]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data["Price"] = pd.to_numeric(
        data["Price"].astype(str).str.replace("K", "", regex=False)
    ) * 1000
    data = data.loc[~data["House_ID"].between(102, 132)]

    transformer = ColumnTransformer(
        [("neighborhood", OneHotEncoder(handle_unknown="ignore"), ["Neighborhood"])],
        remainder="passthrough",
    )
    model = make_pipeline(
        transformer,
        RandomForestRegressor(n_estimators=200, random_state=42),
    )
    model.fit(data[FEATURE_COLUMNS], data["Price"])
    neighborhoods = sorted(data["Neighborhood"].dropna().unique().tolist())
    return model, neighborhoods


try:
    model, neighborhoods = load_model()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Unable to load the housing data: {error}")
    st.stop()

st.markdown('<div class="eyebrow">Property Desk / Customer Intake</div>',
            unsafe_allow_html=True)
st.title("Home price estimate")
st.markdown(
    '<p class="intro">Enter the customer and property details to prepare a local inquiry and estimate.</p>',
    unsafe_allow_html=True,
)

with st.form("customer_intake", clear_on_submit=False):
    customer_col, property_col = st.columns(2, gap="large")
    with customer_col:
        st.markdown(
            '<div class="section-title">Customer details</div>', unsafe_allow_html=True)
        customer_name = st.text_input("Customer name", placeholder="Full name")
        customer_email = st.text_input(
            "Email address", placeholder="name@example.com")
        customer_phone = st.text_input("Phone number", placeholder="Optional")
    with property_col:
        st.markdown(
            '<div class="section-title">Property details</div>', unsafe_allow_html=True)
        square_feet = st.number_input(
            "Square feet", min_value=500, max_value=10000, value=1500, step=50
        )
        bedrooms = st.selectbox("Bedrooms", options=list(range(1, 6)), index=2)
        neighborhood = st.selectbox("Neighborhood", options=neighborhoods)

    submitted = st.form_submit_button(
        "Save inquiry and estimate", type="primary", use_container_width=True)

if submitted:
    if not customer_name.strip():
        st.error("Enter the customer's name to continue.")
    elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", customer_email.strip()):
        st.error("Enter a valid email address.")
    else:
        property_details = pd.DataFrame(
            [{
                "Square_Feet": square_feet,
                "Bedrooms": bedrooms,
                "Neighborhood": neighborhood,
            }]
        )
        estimate = float(model.predict(property_details)[0])
        submission = {
            "Submitted_At": datetime.now().astimezone().isoformat(timespec="seconds"),
            "Customer_Name": customer_name.strip(),
            "Email": customer_email.strip(),
            "Phone": customer_phone.strip(),
            "Square_Feet": square_feet,
            "Bedrooms": bedrooms,
            "Neighborhood": neighborhood,
            "Estimated_Price": round(estimate, 2),
        }
        try:
            pd.DataFrame([submission]).to_csv(
                SUBMISSIONS_PATH,
                mode="a",
                header=not SUBMISSIONS_PATH.exists(),
                index=False,
            )
            st.session_state["last_submission"] = submission
        except OSError as error:
            st.error(f"The inquiry could not be saved: {error}")

if "last_submission" in st.session_state:
    last_submission = st.session_state["last_submission"]
    st.divider()
    st.markdown('<div class="estimate-rule">', unsafe_allow_html=True)
    st.markdown('<div class="estimate-label">Estimated market price</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="estimate-value">${last_submission["Estimated_Price"]:,.0f}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        f"Inquiry saved for {last_submission['Customer_Name']} in {last_submission['Neighborhood']}. "
        "This data-based estimate is not a formal appraisal."
    )

st.caption(
    f"Submitted inquiries are stored locally in `{SUBMISSIONS_PATH.name}`.")
