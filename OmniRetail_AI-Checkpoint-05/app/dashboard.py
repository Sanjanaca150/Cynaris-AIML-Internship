import requests
import pandas as pd
import plotly.express as px
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="OmniRetail AI Manager Dashboard",
    page_icon="📊",
    layout="wide",
)


st.title("📊 OmniRetail AI")
st.subheader("Manager Dashboard")
st.caption(
    "Demand Forecasting • Dynamic Pricing • Inventory Replenishment"
)


def get_api_data(endpoint):
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


try:
    dashboard = get_api_data("/dashboard-data")
    pricing_data = get_api_data("/pricing")
    inventory_data = get_api_data("/inventory")

except requests.exceptions.RequestException as exc:
    st.error(
        "Unable to connect to the OmniRetail AI FastAPI backend."
    )
    st.info(
        "Make sure the FastAPI server is running on "
        "http://127.0.0.1:8000"
    )
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

st.markdown("### Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total SKUs",
        dashboard["total_skus"],
    )

with col2:
    st.metric(
        "Forecast MAPE",
        f'{dashboard["forecast_mape"]}%',
        delta="Target < 15%",
    )

with col3:
    st.metric(
        "Pricing Tested",
        dashboard["pricing_skus_tested"],
        delta="20 SKUs",
    )

with col4:
    st.metric(
        "Inventory Analyzed",
        dashboard["inventory_skus_analyzed"],
        delta="20 SKUs",
    )


st.divider()


# ---------------------------------------------------------
# INVENTORY SUMMARY
# ---------------------------------------------------------

st.markdown("### Inventory Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Healthy",
        dashboard["healthy_inventory"],
    )

with col2:
    st.metric(
        "Reorder Required",
        dashboard["reorder_inventory"],
    )

with col3:
    st.metric(
        "Critical",
        dashboard["critical_inventory"],
    )


inventory_df = pd.DataFrame(
    inventory_data["alerts"]
)

status_counts = (
    inventory_df["status"]
    .value_counts()
    .reset_index()
)

status_counts.columns = [
    "status",
    "count",
]

fig_inventory = px.bar(
    status_counts,
    x="status",
    y="count",
    title="Inventory Status Distribution",
    labels={
        "status": "Inventory Status",
        "count": "Number of SKUs",
    },
)

st.plotly_chart(
    fig_inventory,
    use_container_width=True,
)


st.divider()


# ---------------------------------------------------------
# PRICING SUMMARY
# ---------------------------------------------------------

st.markdown("### Dynamic Pricing")

pricing_df = pd.DataFrame(
    pricing_data["recommendations"]
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Base Price",
        f'₹{dashboard["average_base_price_inr"]:,.2f}',
    )

with col2:
    st.metric(
        "Average Recommended Price",
        f'₹{dashboard["average_recommended_price_inr"]:,.2f}',
    )


pricing_chart_df = pricing_df[
    [
        "sku_id",
        "base_price_inr",
        "recommended_price_inr",
    ]
].copy()

pricing_chart_df = pricing_chart_df.melt(
    id_vars="sku_id",
    value_vars=[
        "base_price_inr",
        "recommended_price_inr",
    ],
    var_name="price_type",
    value_name="price_inr",
)

pricing_chart_df["price_type"] = (
    pricing_chart_df["price_type"]
    .replace(
        {
            "base_price_inr": "Base Price",
            "recommended_price_inr": "Recommended Price",
        }
    )
)

fig_pricing = px.bar(
    pricing_chart_df,
    x="sku_id",
    y="price_inr",
    color="price_type",
    barmode="group",
    title="Base Price vs Recommended Price",
    labels={
        "sku_id": "SKU",
        "price_inr": "Price (INR)",
        "price_type": "Price Type",
    },
)

st.plotly_chart(
    fig_pricing,
    use_container_width=True,
)


st.markdown("#### Pricing Recommendations")

st.dataframe(
    pricing_df,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ---------------------------------------------------------
# INVENTORY ALERT TABLE
# ---------------------------------------------------------

st.markdown("### Inventory Replenishment Alerts")

display_inventory = inventory_df[
    [
        "sku_id",
        "category",
        "inventory",
        "forecast_demand",
        "status",
        "reorder_quantity",
    ]
].copy()

st.dataframe(
    display_inventory,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# ---------------------------------------------------------
# DEMAND FORECAST
# ---------------------------------------------------------

st.markdown("### Demand Forecast")

sku_options = sorted(
    pricing_df["sku_id"].unique()
)

selected_sku = st.selectbox(
    "Select SKU",
    sku_options,
    index=0,
)


if st.button(
    "Generate Forecast",
    type="primary",
):
    try:
        forecast_data = get_api_data(
            f"/forecast/{selected_sku}"
        )

        forecast_df = pd.DataFrame(
            forecast_data["forecast"]
        )

        forecast_df["date"] = pd.to_datetime(
            forecast_df["date"]
        )

        fig_forecast = px.line(
            forecast_df,
            x="date",
            y="predicted_demand",
            title=(
                f"60-Day Demand Forecast — "
                f"{selected_sku}"
            ),
            labels={
                "date": "Date",
                "predicted_demand": "Predicted Demand",
            },
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "SKU",
                forecast_data["sku_id"],
            )

        with col2:
            st.metric(
                "MAPE",
                f'{forecast_data["mape"]}%',
            )

        with col3:
            st.metric(
                "Forecast Days",
                forecast_data["forecast_days"],
            )

    except requests.exceptions.RequestException as exc:
        st.error("Forecast API request failed.")
        st.exception(exc)


st.divider()


# ---------------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------------

st.markdown("### System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("Demand Forecasting: Operational")

with col2:
    st.success("Dynamic Pricing: Operational")

with col3:
    st.success("Inventory Replenishment: Operational")


st.caption(
    "OmniRetail AI • Checkpoint 5 • "
    "FastAPI + Prophet + Random Forest + Streamlit + Plotly"
)