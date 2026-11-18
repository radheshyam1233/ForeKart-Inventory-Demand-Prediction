from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='ForeKart | Inventory Intelligence', page_icon='📦', layout='wide')
ROOT = Path(__file__).parent
DATA = ROOT / 'data'
TARGETS = {'stockout_rate': 10.0, 'excess_stock_unit_days': 15.0, 'rush_order_cost': 20.0, 'inventory_expense': 25.0}
FILES = {
    'sales': 'forekart_cleaned.csv',
    'arima': 'forekart_all_arima_forecasts.csv',
    'rf': 'forekart_rf_forecasts.csv',
    'inventory': 'forekart_inventory_simulation.csv',
    'kpis': 'forekart_policy_kpis.csv',
    'improvements': 'forekart_kpi_improvements.csv',
    'channel': 'forekart_channel_summary.csv',
    'product_channel': 'forekart_product_channel_kpis.csv',
    'stock': 'forekart_stock_coverage.csv',
    'promotions': 'forekart_promotion_recommendations.csv',
    'scenarios': 'forekart_promotion_scenarios.csv',
}
@st.cache_data(show_spinner=False)
def load_file(name):
    path = DATA / FILES[name]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'policy' not in df.columns and name == 'kpis':
        first = df.columns[0]
        if first.startswith('Unnamed') or first == 'policy':
            df = df.rename(columns={first: 'policy'})
    return df

def download_frame(df, name):
    st.download_button('Download CSV', df.to_csv(index=False).encode('utf-8'), file_name=name, mime='text/csv')

def filtered(df, product, channel):
    if df is None:
        return None
    out = df.copy()
    if product != 'All' and 'product' in out:
        out = out[out['product'] == product]
    if channel != 'All' and 'channel' in out:
        out = out[out['channel'] == channel]
    return out

def note_missing(name):
    st.info(f'Missing data/{FILES[name]}. Upload your Module CSV into the data folder and redeploy.')

st.title('📦 ForeKart Inventory Intelligence')
st.caption('Synthetic portfolio case study • Forecasting, replenishment, channel analysis and promotion scenarios • Not verified company savings')
st.warning('Targets are not achievements. The Module 7 simulation requires an audited rerun before inventory improvement percentages can be treated as validated. Promotion elasticity and fees are assumed.')
loaded = {name: load_file(name) for name in FILES}
all_products = sorted({str(p) for key in ('sales', 'inventory', 'arima', 'rf') if loaded[key] is not None and 'product' in loaded[key] for p in loaded[key]['product'].dropna().unique()})
all_channels = sorted({str(c) for key in ('sales', 'inventory', 'arima', 'rf') if loaded[key] is not None and 'channel' in loaded[key] for c in loaded[key]['channel'].dropna().unique() if str(c) != 'Unknown'})
with st.sidebar:
    st.header('Explore the project')
    product = st.selectbox('Product', ['All'] + all_products)
    channel = st.selectbox('Channel', ['All'] + all_channels)
    section = st.radio('Dashboard section', ['Overview', 'Forecasting', 'Inventory Optimization', 'Multi-Channel Analysis', 'Promotions', 'Data Quality & Downloads'])
    st.caption('Filters apply to charts and tables with product/channel columns. Overall policy targets always use the full unfiltered evaluation.')

sales = filtered(loaded['sales'], product, channel)
inv = filtered(loaded['inventory'], product, channel)

if section == 'Overview':
    st.header('Project overview')
    st.markdown('**Resume alignment:** (1) ARIMA + Random Forest forecasts; (2) inventory KPIs and operational costs; (3) multi-channel demand and targeted promotion scenarios.')
    cols = st.columns(4)
    labels = [('Stockout reduction', 10), ('Overstock reduction', 15), ('Rush-order cost reduction', 20), ('Inventory expense reduction', 25)]
    for col, (label, value) in zip(cols, labels):
        col.metric(label + ' target', f'{value}%')
    if sales is not None and {'customer_demand','units_sold','revenue'}.issubset(sales.columns):
        cols = st.columns(3)
        cols[0].metric('Historical customer demand', f'{sales.customer_demand.sum():,.0f} units')
        cols[1].metric('Historical units sold', f'{sales.units_sold.sum():,.0f} units')
        cols[2].metric('Historical revenue', f'₹{sales.revenue.sum():,.0f}')
    else:
        note_missing('sales')
    if loaded['improvements'] is not None:
        df = loaded['improvements'].copy()
        required = {'policy','metric','reduction_percent','target_percent'}
        if required.issubset(df.columns):
            st.subheader('Recorded Module 7 results — provisional until audit')
            st.dataframe(df[list(required)].sort_values(['policy','metric']), use_container_width=True, hide_index=True)
        else:
            st.info('Improvement file columns do not match expected format.')
    else:
        st.info('No measured improvements loaded. Only targets are displayed.')

elif section == 'Forecasting':
    st.header('ARIMA vs Random Forest')
    a, r = filtered(loaded['arima'], product, channel), filtered(loaded['rf'], product, channel)
    if a is None or r is None:
        if a is None: note_missing('arima')
        if r is None: note_missing('rf')
    elif {'date','product','channel','predicted_demand'}.issubset(a.columns) and {'date','product','channel','predicted_demand'}.issubset(r.columns):
        keys = ['date','product','channel']
        comp = a[keys+['predicted_demand']].rename(columns={'predicted_demand':'ARIMA'}).merge(
            r[keys+['predicted_demand']].rename(columns={'predicted_demand':'Random Forest'}), on=keys, how='inner', validate='one_to_one')
        if sales is not None and 'customer_demand' in sales:
            comp = comp.merge(sales[keys+['customer_demand']].rename(columns={'customer_demand':'Actual'}), on=keys, how='left', validate='one_to_one')
        elif 'actual_demand' in r:
            comp = comp.merge(r[keys+['actual_demand']].rename(columns={'actual_demand':'Actual'}), on=keys, how='left', validate='one_to_one')
        if comp.empty:
            st.info('No matching product-channel-date observations for these filters.')
        else:
            st.caption('Chart aggregates daily demand across the selected product/channel groups.')
            cols = ['ARIMA','Random Forest'] + (['Actual'] if 'Actual' in comp else [])
            daily = comp.groupby('date')[cols].sum(min_count=1)
            st.line_chart(daily)
            if 'Actual' in comp:
                valid = comp.dropna(subset=['Actual','ARIMA','Random Forest'])
                if len(valid):
                    metrics=[]
                    for model in ['ARIMA','Random Forest']:
                        err = valid['Actual'] - valid[model]
                        metrics.append({'Model':model,'MAE':err.abs().mean(),'RMSE':np.sqrt((err**2).mean()),'Matched observations':len(valid)})
                    st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True)
            st.dataframe(comp.head(200), hide_index=True, use_container_width=True)
            download_frame(comp, 'forekart_matched_forecasts.csv')
    else:
        st.error('Forecast files do not have the expected columns.')

elif section == 'Inventory Optimization':
    st.header('Inventory policy comparison')
    if inv is None:
        note_missing('inventory')
    else:
        st.caption('Daily simulation results. Interpret cost and percentage reductions as provisional until the replenishment and end-of-horizon accounting audit is rerun.')
        required = {'policy','date','stockout','excess_stock','rush_order_cost','inventory_expense'}
        if 'inventory_expense' not in inv and {'purchase_spend','holding_cost','rush_order_cost'}.issubset(inv):
            inv = inv.copy()
            inv['inventory_expense'] = inv['purchase_spend']+inv['holding_cost']+inv['rush_order_cost']
        if required.issubset(inv.columns):
            summary = inv.groupby('policy').agg(
                records=('stockout','size'), stockout_days=('stockout','sum'), excess_unit_days=('excess_stock','sum'),
                rush_order_cost=('rush_order_cost','sum'), inventory_expense=('inventory_expense','sum')).reset_index()
            summary['stockout_rate'] = 100*summary['stockout_days']/summary['records']
            st.dataframe(summary, hide_index=True, use_container_width=True)
            if 'closing_stock' in inv:
                daily = inv.groupby(['date','policy'])['closing_stock'].sum().unstack('policy')
                st.line_chart(daily)
            st.subheader('Audit checks')
            if {'demand','units_sold','unmet_demand'}.issubset(inv.columns):
                consistent = np.isclose(inv['demand'], inv['units_sold']+inv['unmet_demand']).all()
                st.write('Demand = sales + unmet demand:', 'PASS' if consistent else 'FAIL')
                cross = inv.pivot_table(index=['date','product','channel'],columns='policy',values='demand',aggfunc='first')
                if cross.shape[1]>1:
                    st.write('Same demand for all policies:', 'PASS' if cross.nunique(axis=1).eq(1).all() else 'FAIL')
            st.info('These checks cannot detect duplicate normal/rush orders, unrecorded pipeline inventory, or biased expense comparisons. Rerun the corrected simulator before claiming validated savings.')
            download_frame(summary, 'forekart_filtered_inventory_kpis.csv')
        else:
            st.error('Inventory file is missing required KPI columns.')

elif section == 'Multi-Channel Analysis':
    st.header('Channel and product demand')
    if sales is None:
        note_missing('sales')
    elif {'channel','product','customer_demand','units_sold','revenue'}.issubset(sales.columns):
        s = sales[sales['channel'].isin(['Website','Mobile App','Marketplace'])].copy()
        if s.empty:
            st.info('No rows match these filters.')
        else:
            by_channel = s.groupby('channel').agg(demand=('customer_demand','sum'),sold=('units_sold','sum'),revenue=('revenue','sum')).reset_index()
            by_channel['fill_rate_pct'] = np.where(by_channel.demand>0,100*by_channel.sold/by_channel.demand,np.nan)
            st.dataframe(by_channel, hide_index=True, use_container_width=True)
            st.bar_chart(by_channel.set_index('channel')['demand'])
            pivot = s.pivot_table(index='product',columns='channel',values='customer_demand',aggfunc='sum',fill_value=0)
            st.subheader('Product × channel demand')
            st.dataframe(pivot, use_container_width=True)
            if 'date' in s:
                s['month'] = s['date'].dt.to_period('M').dt.to_timestamp()
                monthly = s.groupby(['month','channel'])['customer_demand'].sum().unstack(fill_value=0)
                st.line_chart(monthly)
            download_frame(by_channel, 'forekart_filtered_channel_summary.csv')
    else:
        st.error('Sales file lacks required columns.')

elif section == 'Promotions':
    st.header('Promotion scenarios and recommendations')
    st.caption('Assumed elasticity and costs: exploratory scenarios only, not causal proof or realized savings.')
    stock = filtered(loaded['stock'], product, channel)
    if stock is not None and {'product','channel','excess_stock'}.issubset(stock):
        st.subheader('Excess-stock candidates')
        st.dataframe(stock.sort_values('excess_stock',ascending=False).head(30), hide_index=True, use_container_width=True)
    else:
        note_missing('stock')
    promo = filtered(loaded['promotions'], product, channel)
    if promo is not None:
        st.subheader('Module 8 recommendations')
        st.dataframe(promo, hide_index=True, use_container_width=True)
        download_frame(promo, 'forekart_filtered_promotions.csv')
    else:
        note_missing('promotions')
    scenarios = filtered(loaded['scenarios'], product, channel)
    if scenarios is not None and {'discount_percent','incremental_contribution','response_scenario'}.issubset(scenarios):
        st.subheader('Scenario sensitivity')
        view = scenarios.groupby(['discount_percent','response_scenario'])['incremental_contribution'].mean().unstack()
        st.line_chart(view)

else:
    st.header('Data quality and downloads')
    st.write('Loaded files and available records:')
    for key, filename in FILES.items():
        df = loaded[key]
        st.write(('✅' if df is not None else '⚠️'), filename, f'— {len(df):,} rows' if df is not None else '— not supplied')
    st.markdown('**Data provenance:** The case study uses synthetic demand and illustrative inventory/promotion assumptions. Model performance and simulated improvements must be evaluated from actual exported outputs.')
    st.markdown('**Audit checklist:** chronological train/test split; no target leakage; identical test demand; consistent lead times; no duplicate replenishment; outstanding orders accounted for; target claims supported by measured reductions; elasticity not misrepresented as causal.')
    for key, df in loaded.items():
        if df is not None:
            with st.expander(f'Download {FILES[key]}'):
                download_frame(df, FILES[key])
