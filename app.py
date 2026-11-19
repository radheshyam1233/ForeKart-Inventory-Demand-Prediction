"""ForeKart | Demand Forecasting & Inventory Analytics."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ForeKart | Inventory Intelligence", page_icon="📦", layout="wide")
DATA = Path(__file__).resolve().parent / "data"

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1250px}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:#f4f7fb;border:1px solid #e5eaf1;border-radius:12px;padding:14px}
@media(prefers-color-scheme:dark){[data-testid="stMetric"]{background:#17202c;border-color:#354052}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(name):
    path = DATA / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Unable to read {name}: {exc}")
        return None


def note_missing(names):
    st.info("To display this section, add the following CSV file(s): " + ", ".join(f"`{name}`" for name in names))


def forecast_frame(frame, prediction_name):
    """Normalize forecast keys and values; average duplicate keys rather than multiplying rows on merge."""
    keys = ["date", "product", "channel"]
    result = frame[keys + ["predicted_demand"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["product"] = result["product"].astype("string").str.strip()
    result["channel"] = result["channel"].astype("string").str.strip()
    result["predicted_demand"] = pd.to_numeric(result["predicted_demand"], errors="coerce")
    result = result.dropna(subset=keys + ["predicted_demand"])
    return result.groupby(keys, as_index=False, dropna=False)["predicted_demand"].mean().rename(columns={"predicted_demand": prediction_name})


def error_metrics(actual, predicted):
    valid = pd.DataFrame({"actual": actual, "predicted": predicted}).apply(pd.to_numeric, errors="coerce").dropna()
    if valid.empty:
        return None
    errors = valid["actual"] - valid["predicted"]
    return {"MAE": errors.abs().mean(), "RMSE": np.sqrt((errors ** 2).mean()), "n": len(valid)}


sales = load_csv("forekart_cleaned.csv")
comparison = load_csv("forekart_model_comparison.csv")
arima = load_csv("forekart_all_arima_forecasts.csv")
rf_forecasts = load_csv("forekart_rf_forecasts.csv")
summary = load_csv("forekart_policy_kpis_final.csv")
improvements = load_csv("forekart_kpi_improvements_final.csv")
tuning = load_csv("forekart_inventory_tuning.csv")
channel_summary = load_csv("forekart_channel_summary.csv")

st.title("📦 ForeKart | Inventory Demand Prediction")
st.caption("ARIMA + Random Forest • Inventory policy simulation • Multi-channel analysis")
st.info("**Portfolio demonstration:** This project uses synthetic data. Forecast evaluation and inventory outcomes are not verified company results.")

tab1, tab2, tab3 = st.tabs(["📈 Demand Forecasting", "📦 Inventory Optimization", "🛍️ Channels & Promotions"])

with tab1:
    st.header("Demand forecasting | Resume bullet 1")
    st.write("Compare ARIMA and Random Forest predictions for a selected product and sales channel.")

    if comparison is not None and {"model", "MAE", "RMSE"}.issubset(comparison.columns):
        metrics = comparison[["model", "MAE", "RMSE"]].copy()
        metrics["MAE"] = pd.to_numeric(metrics["MAE"], errors="coerce")
        metrics["RMSE"] = pd.to_numeric(metrics["RMSE"], errors="coerce")
        st.subheader("Forecast accuracy (lower is better)")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption("MAE and RMSE are in demand units. ARIMA has the lower MAE in this export; Random Forest has the lower RMSE. The best choice depends on the error measure and business cost of errors.")
    else:
        note_missing(["forekart_model_comparison.csv"])

    required = {"date", "product", "channel", "predicted_demand"}
    if arima is not None and rf_forecasts is not None and required.issubset(arima.columns) and required.issubset(rf_forecasts.columns):
        a = forecast_frame(arima, "ARIMA")
        r = forecast_frame(rf_forecasts, "Random Forest")
        forecast = a.merge(r, on=["date", "product", "channel"], how="inner", validate="one_to_one")

        actual_source = arima if "actual_demand" in arima.columns else rf_forecasts if "actual_demand" in rf_forecasts.columns else None
        if actual_source is not None:
            actual = actual_source[["date", "product", "channel", "actual_demand"]].copy()
            actual["date"] = pd.to_datetime(actual["date"], errors="coerce")
            actual["product"] = actual["product"].astype("string").str.strip()
            actual["channel"] = actual["channel"].astype("string").str.strip()
            actual["actual_demand"] = pd.to_numeric(actual["actual_demand"], errors="coerce")
            actual = actual.dropna(subset=["date", "product", "channel", "actual_demand"])
            actual = actual.groupby(["date", "product", "channel"], as_index=False)["actual_demand"].mean()
            forecast = forecast.merge(actual, on=["date", "product", "channel"], how="inner", validate="one_to_one")

        products = sorted(forecast["product"].dropna().astype(str).unique())
        channels = sorted(forecast["channel"].dropna().astype(str).unique())
        if products and channels:
            c1, c2 = st.columns(2)
            product = c1.selectbox("Product", products)
            channel = c2.selectbox("Sales channel", channels)
            selected = forecast.loc[(forecast["product"] == product) & (forecast["channel"] == channel)].sort_values("date").copy()
            if not selected.empty:
                start_date, end_date = selected["date"].min(), selected["date"].max()
                st.subheader("Daily demand: actual vs predicted")
                st.caption(f"Product: {product} · Channel: {channel} · Period: {start_date:%b %d, %Y}–{end_date:%b %d, %Y} · {len(selected):,} matched daily observations")
                st.caption("Horizontal axis: Date · Vertical axis: Demand (units per day)")
                chart_data = selected.set_index("date")[["actual_demand", "ARIMA", "Random Forest"]].rename(columns={"actual_demand": "Actual demand"})
                st.line_chart(chart_data, height=380)

                st.markdown("**Selected-series evaluation errors**")
                metric_cols = st.columns(2)
                for col, model in zip(metric_cols, ["ARIMA", "Random Forest"]):
                    result = error_metrics(selected["actual_demand"], selected[model])
                    if result:
                        col.metric(f"{model} MAE", f"{result['MAE']:.2f} units")
                        col.caption(f"RMSE: {result['RMSE']:.2f} units · n = {result['n']:,} matched dates")
                st.caption("The forecast files pair predictions with actual demand, so this is a historical evaluation comparison—not a future forecast. The exports do not identify a train/test boundary; confirm the time-based split in the model-generation code before describing these as held-out test results.")
            else:
                st.info("No matching forecast rows for this product and channel.")
        else:
            st.info("No valid matching forecast rows were found after cleaning dates and duplicate keys.")
    elif arima is None or rf_forecasts is None:
        note_missing([name for name, frame in [("forekart_all_arima_forecasts.csv", arima), ("forekart_rf_forecasts.csv", rf_forecasts)] if frame is None])
    else:
        st.warning("Forecast CSVs are missing one or more required columns: date, product, channel, predicted_demand.")

with tab2:
    st.header("Inventory optimization | Resume bullet 2")
    st.write("Compare replenishment-policy simulations against the project goals.")

    if tuning is not None and {"safety_z", "cover_days"}.issubset(tuning.columns) and not tuning.empty:
        valid_tuning = tuning.copy()
        if "operating_cost" in valid_tuning:
            valid_tuning["operating_cost"] = pd.to_numeric(valid_tuning["operating_cost"], errors="coerce")
            valid_tuning = valid_tuning.dropna(subset=["operating_cost"])
        if not valid_tuning.empty:
            best = valid_tuning.sort_values("operating_cost").iloc[0] if "operating_cost" in valid_tuning.columns else valid_tuning.iloc[0]
            st.caption(f"Lowest-cost setting in the exported tuning results: safety Z = {best['safety_z']:g}; coverage = {int(best['cover_days'])} days. Selected on validation data; not an independently verified optimum.")

    st.subheader("Project target KPIs")
    target_kpis = pd.DataFrame([
        {"Target metric": "Stockouts", "Target change": "10% reduction"},
        {"Target metric": "Overstock", "Target change": "15% reduction"},
        {"Target metric": "Rush-order costs", "Target change": "20% reduction"},
        {"Target metric": "Inventory expenses", "Target change": "25% reduction"},
    ])
    target_cols = st.columns(4)
    for col, row in zip(target_cols, target_kpis.to_dict("records")):
        col.metric(row["Target metric"], row["Target change"])
    st.caption("These percentages are the project's stated targets from the resume requirement. They are not represented as achieved simulation results.")

    if improvements is not None and "policy" in improvements.columns:
        with st.expander("Raw simulation outputs (separate from project targets)"):
            improvement_cols = {
                "stockout_reduction_pct": "Stockout reduction (%)",
                "overstock_reduction_pct": "Overstock reduction (%)",
                "rush_cost_reduction_pct": "Rush-order cost reduction (%)",
                "inventory_cost_reduction_pct": "Inventory cost reduction (%)",
            }
            available = [column for column in improvement_cols if column in improvements.columns]
            if available:
                view = improvements[["policy"] + available].copy().rename(columns=improvement_cols)
                for column in view.columns[1:]:
                    view[column] = pd.to_numeric(view[column], errors="coerce")
                st.dataframe(view, use_container_width=True, hide_index=True)
                st.caption("Raw values from the exported synthetic simulation. Blank values are unavailable; negative reductions mean the metric increased versus baseline. These are not the project targets.")
            else:
                st.warning("The improvement file has no recognized KPI columns.")
    else:
        st.info("Raw simulation improvement output is unavailable; project targets above are still displayed.")

    if summary is not None and "policy" in summary.columns:
        st.subheader("Policy comparison")
        cols = ["policy", "total_demand", "units_sold", "stockout_records", "fill_rate", "average_stock", "rush_order_cost", "inventory_carrying_ordering_cost", "operating_cost"]
        display = summary[[c for c in cols if c in summary.columns]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Use the definitions and assumptions from the simulation code when interpreting fill rate, stockouts, and operating cost. The README documents known simulation limitations that should be resolved before treating these as decision-grade results.")
    else:
        note_missing(["forekart_policy_kpis_final.csv"])

with tab3:
    st.header("Multi-channel demand & promotions | Resume bullet 3")
    st.write("Compare observed demand across Website, Mobile App, and Marketplace channels, and inspect promotional versus non-promotional records.")
    if sales is not None and {"channel", "customer_demand", "units_sold"}.issubset(sales.columns):
        df = sales.copy()
        df["customer_demand"] = pd.to_numeric(df["customer_demand"], errors="coerce")
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
        df = df.dropna(subset=["channel", "customer_demand", "units_sold"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        revenue_available = "revenue" in df.columns or "selling_price" in df.columns
        if "revenue" in df.columns:
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        elif "selling_price" in df.columns:
            df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
            df["revenue"] = df["units_sold"] * df["selling_price"]

        c1, c2 = st.columns(2)
        c1.metric("Customer demand (units)", f"{df['customer_demand'].sum():,.0f}")
        c2.metric("Units sold", f"{df['units_sold'].sum():,.0f}")
        if revenue_available:
            st.metric("Synthetic revenue", f"{df['revenue'].sum(min_count=1):,.2f}", help="Calculated from supplied revenue or units sold × selling price; not verified company revenue.")

        group_metrics = {"customer_demand": "sum", "units_sold": "sum"}
        if revenue_available:
            group_metrics["revenue"] = "sum"
        by_channel = df.groupby("channel", as_index=False).agg(**{name: (name, agg) for name, agg in group_metrics.items()})
        by_channel = by_channel.rename(columns={"customer_demand": "demand"})
        st.subheader("Channel performance")
        st.bar_chart(by_channel.set_index("channel")[["demand", "units_sold"]])
        st.dataframe(by_channel, use_container_width=True, hide_index=True)

        if "product" in df.columns:
            st.subheader("Product demand by channel")
            pivot = df.pivot_table(index="product", columns="channel", values="customer_demand", aggfunc="sum", fill_value=0)
            st.bar_chart(pivot)

        if "promotion" in df.columns:
            st.subheader("Promotion patterns")
            raw_promo = df["promotion"].astype("string").str.strip().str.lower()
            promo_map = {"1": True, "true": True, "yes": True, "y": True, "0": False, "false": False, "no": False, "n": False}
            df["promotion_group"] = raw_promo.map(promo_map)
            promo_df = df.dropna(subset=["promotion_group"])
            if not promo_df.empty:
                promo = promo_df.groupby(["channel", "promotion_group"], as_index=False).agg(
                    observations=("customer_demand", "size"),
                    avg_demand=("customer_demand", "mean"),
                    avg_units_sold=("units_sold", "mean"),
                )
                promo["promotion_group"] = promo["promotion_group"].map({True: "Promotion", False: "No promotion"})
                st.dataframe(promo.rename(columns={"promotion_group": "promotion_status"}), use_container_width=True, hide_index=True)
            else:
                st.info("No recognized promotion values (true/false, yes/no, or 1/0) were found.")
            st.caption("These are observational comparisons in synthetic data, not evidence that promotions caused a change in demand. Unrecognized or missing promotion values are excluded.")
        else:
            st.info("Promotion field not found in the sales dataset.")
    elif channel_summary is not None:
        st.dataframe(channel_summary, use_container_width=True, hide_index=True)
        st.info("Add `forekart_cleaned.csv` to unlock promotion-level comparisons.")
    else:
        note_missing(["forekart_cleaned.csv"])

st.divider()
st.caption("ForeKart | Portfolio analytics • Synthetic sales records • Forecast and simulation outcomes depend on data and assumptions; do not present them as real company performance.")
