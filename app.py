"""ForeKart | Demand Forecasting & Inventory Analytics.

Place exported ForeKart CSV files in ./data/ before running:
    streamlit run app.py
All inventory outcomes are from synthetic-data simulations.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ForeKart | Inventory Intelligence", page_icon="📦", layout="wide")
DATA = Path(__file__).resolve().parent / "data"

st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1250px}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:#f4f7fb;border:1px solid #e5eaf1;border-radius:12px;padding:14px}
@media(prefers-color-scheme:dark){[data-testid="stMetric"]{background:#17202c;border-color:#354052}}
</style>""", unsafe_allow_html=True)

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

def first_col(df, *names):
    if df is None:
        return None
    return next((name for name in names if name in df.columns), None)

def fmt_num(value, decimals=0):
    if pd.isna(value):
        return "Not available"
    return f"{value:,.{decimals}f}"

def metric_pct(row, key):
    if row is None or key not in row or pd.isna(row[key]):
        return "Not verified"
    return f"{float(row[key]):+.2f}%"

def note_missing(names):
    st.info("To display this section, add the following CSV file(s) to the repository's `data/` folder: " + ", ".join(f"`{n}`" for n in names))

sales = load_csv("forekart_cleaned.csv")
comparison = load_csv("forekart_model_comparison.csv")
arima = load_csv("forekart_all_arima_forecasts.csv")
rf_forecasts = load_csv("forekart_rf_forecasts.csv")
summary = load_csv("forekart_policy_kpis_final.csv")
improvements = load_csv("forekart_kpi_improvements_final.csv")
tuning = load_csv("forekart_inventory_tuning.csv")
channel_summary = load_csv("forekart_channel_summary.csv")

st.title("📦 ForeKart | Inventory Demand Prediction")
st.caption("ARIMA + Random Forest • Inventory policy optimization • Multi-channel insights")
st.info("**Portfolio demonstration:** Data is synthetic, and inventory improvements are simulated—not observed company results. Resume percentages are targets unless independently supported by the analysis.")

tab1, tab2, tab3 = st.tabs(["📈 Demand Forecasting", "📦 Inventory Optimization", "🛍️ Channels & Promotions"])

with tab1:
    st.header("Demand forecasting | Resume bullet 1")
    st.write("Compare ARIMA and Random Forest forecasts across six products and three sales channels. The inventory goals are **10% fewer stockouts** and **15% less overstock**.")
    if comparison is not None and {"model", "MAE", "RMSE"}.issubset(comparison.columns):
        metrics = comparison[["model", "MAE", "RMSE"]].copy()
        st.subheader("Forecast accuracy (lower is better)")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.bar_chart(metrics.set_index("model")[["MAE", "RMSE"]])
    else:
        note_missing(["forekart_model_comparison.csv"])

    if arima is not None and rf_forecasts is not None:
        needed = {"date", "product", "channel", "predicted_demand"}
        if needed.issubset(arima.columns) and needed.issubset(rf_forecasts.columns):
            products = sorted(set(arima["product"].dropna().astype(str)) & set(rf_forecasts["product"].dropna().astype(str)))
            channels = sorted(set(arima["channel"].dropna().astype(str)) & set(rf_forecasts["channel"].dropna().astype(str)))
            if products and channels:
                c1, c2 = st.columns(2)
                product = c1.selectbox("Product", products)
                channel = c2.selectbox("Sales channel", channels)
                cols = ["date", "product", "channel", "predicted_demand"]
                a = arima[cols].rename(columns={"predicted_demand": "ARIMA"})
                r = rf_forecasts[cols].rename(columns={"predicted_demand": "Random Forest"})
                selected = a.merge(r, on=["date", "product", "channel"], how="inner")
                actual_col = first_col(arima, "actual_demand")
                if actual_col:
                    actual = arima[["date", "product", "channel", actual_col]].rename(columns={actual_col: "Actual demand"})
                    selected = selected.merge(actual, on=["date", "product", "channel"], how="left")
                selected = selected[(selected["product"].astype(str) == product) & (selected["channel"].astype(str) == channel)].copy()
                selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
                selected = selected.dropna(subset=["date"]).sort_values("date")
                if not selected.empty:
                    st.subheader("Daily demand: actual vs predicted")
                    st.line_chart(selected.set_index("date")[[c for c in ["Actual demand", "ARIMA", "Random Forest"] if c in selected.columns]])
                else:
                    st.info("No matching forecast rows for this product and channel.")
        else:
            st.warning("Forecast files do not contain the expected date, product, channel and predicted_demand columns.")
    else:
        note_missing(["forekart_all_arima_forecasts.csv", "forekart_rf_forecasts.csv"])

with tab2:
    st.header("Inventory optimization | Resume bullet 2")
    st.write("Compare policies using validation-selected safety stock and coverage settings. **Targets:** 20% lower rush-order costs and 25% lower inventory expenses.")
    if tuning is not None and {"safety_z", "cover_days"}.issubset(tuning.columns) and not tuning.empty:
        best = tuning.sort_values("operating_cost").iloc[0] if "operating_cost" in tuning.columns else tuning.iloc[0]
        st.caption(f"Validation-selected parameters: safety Z = {best['safety_z']:g}; coverage = {int(best['cover_days'])} days. These settings were selected using a historical-average baseline, not separately optimized for each forecast model.")

    if improvements is not None and "policy" in improvements.columns:
        policies = improvements["policy"].dropna().tolist()
        if policies:
            selected_policy = st.selectbox("Policy", policies, index=policies.index("Random Forest") if "Random Forest" in policies else 0)
            row = improvements.loc[improvements["policy"] == selected_policy].iloc[0]
            labels = [
                ("Stockout reduction", "10% target", "stockout_reduction_pct"),
                ("Overstock reduction", "15% target", "overstock_reduction_pct"),
                ("Rush-order cost reduction", "20% target", "rush_cost_reduction_pct"),
                ("Inventory expense reduction", "25% target", "inventory_cost_reduction_pct"),
            ]
            cols = st.columns(4)
            for col, (label, target, key) in zip(cols, labels):
                with col:
                    if key == "inventory_cost_reduction_pct":
                        st.metric(label, "Not achieved", help="The measured carrying-and-ordering cost increased; this is not a verified reduction in total inventory expenses.")
                    elif key == "overstock_reduction_pct":
                        st.metric(label, "Not verified", help="The independent excess-stock baseline was zero under the selected benchmark.")
                    else:
                        st.metric(label, metric_pct(row, key), help=target + "; measured on synthetic-data simulation")
                    st.caption(target)
            st.caption("The final inventory_cost_reduction_pct field measures **carrying and ordering costs**, not accounting inventory expenses. Negative percentages mean higher costs. Overstock is unavailable because the independent benchmark returned zero excess units for the baseline.")
            st.subheader("Measured reductions vs baseline")
            plot_data = pd.DataFrame({
                "Metric": ["Stockout events", "Rush-order fees", "Carrying + ordering costs"],
                "Reduction (%)": [row.get("stockout_reduction_pct"), row.get("rush_cost_reduction_pct"), row.get("inventory_cost_reduction_pct")],
            }).set_index("Metric")
            st.bar_chart(plot_data)
        else:
            st.warning("The improvement file has no policy rows.")
    else:
        note_missing(["forekart_kpi_improvements_final.csv"])

    if summary is not None and "policy" in summary.columns:
        st.subheader("Policy comparison")
        cols = ["policy", "total_demand", "units_sold", "stockout_records", "fill_rate", "average_stock", "rush_order_cost", "inventory_carrying_ordering_cost", "operating_cost"]
        st.dataframe(summary[[c for c in cols if c in summary.columns]], use_container_width=True, hide_index=True)
        st.caption("Operating cost includes simulated holding, ordering and lost-sale penalties. Procurement spending is a separate cash-flow measure, not an expense reduction claim.")
    else:
        note_missing(["forekart_policy_kpis_final.csv"])

with tab3:
    st.header("Multi-channel demand & promotions | Resume bullet 3")
    st.write("Explore Website, Mobile App and Marketplace demand, then examine observed associations between promotions and demand in the synthetic dataset.")
    if sales is not None and {"channel", "customer_demand", "units_sold"}.issubset(sales.columns):
        df = sales.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.NaT
        if "revenue" not in df.columns:
            df["revenue"] = df["units_sold"] * df["selling_price"] if "selling_price" in df.columns else 0
        total_demand = df["customer_demand"].sum()
        total_sales = df["units_sold"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Demand (units)", fmt_num(total_demand))
        c2.metric("Units sold", fmt_num(total_sales))
        c3.metric("Simulated revenue", fmt_num(df["revenue"].sum(), 0))

        by_channel = df.groupby("channel", as_index=False).agg(demand=("customer_demand", "sum"), units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
        st.subheader("Channel performance")
        st.bar_chart(by_channel.set_index("channel")[["demand", "units_sold"]])
        st.dataframe(by_channel, use_container_width=True, hide_index=True)

        if "product" in df.columns:
            st.subheader("Product demand by channel")
            pivot = df.pivot_table(index="product", columns="channel", values="customer_demand", aggfunc="sum", fill_value=0)
            st.bar_chart(pivot)

        if "promotion" in df.columns:
            st.subheader("Promotion patterns")
            df["promotion_group"] = df["promotion"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
            promo = df.groupby(["channel", "promotion_group"], as_index=False).agg(
                observations=("customer_demand", "size"),
                avg_demand=("customer_demand", "mean"),
                avg_units_sold=("units_sold", "mean"),
            )
            promo["promotion_group"] = promo["promotion_group"].map({True: "Promotion", False: "No promotion"})
            st.dataframe(promo.rename(columns={"promotion_group": "promotion_status"}), use_container_width=True, hide_index=True)
            st.caption("Promotional and non-promotional records are observational comparisons in generated data. They do not establish causal promotional lift, profitability, or campaign ROI.")
            st.markdown("**Suggested business follow-up:** Test targeted offers on channels with strong demand but lower conversion or fulfillment, and evaluate incremental profit with a controlled experiment.")
        else:
            st.info("Promotion field not found in the sales dataset.")
    elif channel_summary is not None:
        st.dataframe(channel_summary, use_container_width=True, hide_index=True)
        st.info("Add `forekart_cleaned.csv` to unlock promotion-level comparisons.")
    else:
        note_missing(["forekart_cleaned.csv"])

st.divider()
st.caption("ForeKart | Portfolio analytics • Synthetic sales records • Simulation outcomes depend on replenishment assumptions and should not be presented as real company performance.")
