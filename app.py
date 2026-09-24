import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Price Guardian",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ AI PRICE GUARDIAN")
st.subheader("Your Budget. Your Location. Your Best Basket.")

st.write(
    "AI Price Guardian uses historical food-price data and "
    "machine learning to provide price prediction and price-increase risk."
)

st.divider()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    linear_model = joblib.load(
        "linear_price_model.joblib"
    )

    logistic_model = joblib.load(
        "logistic_risk_model.joblib"
    )

    return linear_model, logistic_model


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    prices = pd.read_csv(
        "latest_prices.csv"
    )

    items = pd.read_csv(
        "lookup_item.csv"
    )

    premises = pd.read_csv(
        "lookup_premise.csv"
    )

    return prices, items, premises


try:

    linear_model, logistic_model = load_models()

    prices, items, premises = load_data()

except Exception as e:

    st.error("There was a problem loading the model or data files.")

    st.code(str(e))

    st.stop()


# =========================================================
# PREPARE DATA
# =========================================================

prices["date"] = pd.to_datetime(
    prices["date"],
    errors="coerce"
)

# Merge item information
prices = prices.merge(
    items[
        [
            "item_code",
            "item"
        ]
    ],
    on="item_code",
    how="left"
)

# Merge premise information
prices = prices.merge(
    premises[
        [
            "premise_code",
            "state",
            "district",
            "premise_type"
        ]
    ],
    on="premise_code",
    how="left"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🛒 Shopping Input")

# Product selection
product_list = sorted(
    prices["item"].dropna().unique()
)

selected_product = st.sidebar.selectbox(
    "Select food item",
    product_list
)


# District selection
district_list = sorted(
    prices["district"].dropna().unique()
)

selected_district = st.sidebar.selectbox(
    "Select district",
    district_list
)


# Budget
budget = st.sidebar.number_input(
    "Your budget (RM)",
    min_value=1.0,
    value=100.0,
    step=10.0
)


predict_button = st.sidebar.button(
    "🔮 Predict Price",
    type="primary"
)


# =========================================================
# FILTER DATA
# =========================================================

filtered = prices[
    (prices["item"] == selected_product) &
    (prices["district"] == selected_district)
].copy()


if len(filtered) == 0:

    st.warning(
        "No price records were found for this food item "
        "and district."
    )

    st.stop()


# =========================================================
# LATEST RECORD
# =========================================================

filtered = filtered.sort_values(
    "date"
)

latest = filtered.iloc[-1]


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

current_price = float(
    latest["price"]
)

# Use stored previous_price if available
if "previous_price" in latest.index and pd.notna(
    latest["previous_price"]
):

    previous_price = float(
        latest["previous_price"]
    )

else:

    # Fallback if previous_price is not stored
    previous_price = current_price


date_value = latest["date"]

month = int(
    date_value.month
)

year = int(
    date_value.year
)


# =========================================================
# CREATE MODEL INPUT
# =========================================================

model_input = pd.DataFrame({

    "previous_price": [
        previous_price
    ],

    "price": [
        current_price
    ],

    "month": [
        month
    ],

    "year": [
        year
    ],

    "item": [
        latest["item"]
    ],

    "state": [
        latest["state"]
    ],

    "district": [
        latest["district"]
    ],

    "premise_type": [
        latest["premise_type"]
    ]

})


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # Linear Regression
        predicted_price = linear_model.predict(
            model_input
        )[0]

        # Logistic Regression
        increase_probability = logistic_model.predict_proba(
            model_input
        )[0][1]

        increase_prediction = logistic_model.predict(
            model_input
        )[0]


        # Prevent negative predicted price
        predicted_price = max(
            0,
            predicted_price
        )


        # =================================================
        # RESULTS
        # =================================================

        st.header("🤖 AI Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Current Price",
                f"RM {current_price:.2f}"
            )

        with col2:

            st.metric(
                "Predicted Next Price",
                f"RM {predicted_price:.2f}"
            )

        with col3:

            st.metric(
                "Price Increase Risk",
                f"{increase_probability * 100:.1f}%"
            )


        st.divider()


        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader("🔎 What the AI says")

        if increase_prediction == 1:

            st.warning(
                f"The model estimates a "
                f"{increase_probability * 100:.1f}% probability "
                f"that the next observed price will be higher "
                f"than the current price."
            )

        else:

            st.success(
                f"The model estimates a "
                f"{increase_probability * 100:.1f}% probability "
                f"of a price increase in the next observation."
            )


        # =================================================
        # BUDGET CALCULATION
        # =================================================

        current_quantity = int(
            budget // current_price
        )

        predicted_quantity = int(
            budget // predicted_price
        ) if predicted_price > 0 else 0


        st.subheader("💰 Budget Impact")

        b1, b2 = st.columns(2)

        with b1:

            st.metric(
                f"Units you can buy now with RM {budget:.0f}",
                current_quantity
            )

        with b2:

            st.metric(
                f"Units at predicted price",
                predicted_quantity
            )


        # =================================================
        # PRICE COMPARISON
        # =================================================

        st.subheader(
            "🏪 Price Comparison in Selected District"
        )

        comparison = filtered[
            [
                "premise_code",
                "premise_type",
                "price",
                "date"
            ]
        ].copy()

        comparison = comparison.sort_values(
            "price"
        ).drop_duplicates(
            "premise_code"
        )

        comparison = comparison.head(10)

        comparison["price"] = comparison[
            "price"
        ].round(2)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # AI INSIGHT
        # =================================================

        st.subheader("💡 AI Shopping Insight")

        difference = (
            predicted_price -
            current_price
        )

        if difference > 0:

            st.write(
                f"📈 The predicted next price is approximately "
                f"RM {difference:.2f} higher than the current "
                f"price."
            )

        else:

            st.write(
                f"📉 The predicted next price is approximately "
                f"RM {abs(difference):.2f} lower than the current "
                f"price."
            )

        st.write(
            "Use this prediction as decision-support information. "
            "The final purchasing decision remains with the consumer."
        )


    except Exception as e:

        st.error(
            "The model could not make a prediction."
        )

        st.code(str(e))
