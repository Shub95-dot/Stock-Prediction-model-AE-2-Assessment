import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path

# ==============================================================================
# CONFIG & STATE
# ==============================================================================
st.set_page_config(
    page_title="SOLiGence Barometer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Dark Theme CSS Injection to make it look premium
st.markdown("""
<style>
/* Base Dark Theme Styling */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
.metric-card {
    background: linear-gradient(145deg, #161b22, #0d1117);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    text-align: center;
    transition: transform 0.2s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: #58a6ff;
}
.signal-title {
    font-size: 1.1rem;
    color: #8b949e;
    margin-bottom: 0.5rem;
}
.signal-value {
    font-size: 2.2rem;
    font-weight: 700;
}
.signal-BUY { color: #3fb950; text-shadow: 0 0 10px rgba(63,185,80,0.5); }
.signal-SELL { color: #f85149; text-shadow: 0 0 10px rgba(248,81,73,0.5); }
.signal-HOLD { color: #d2a8ff; }
.meta-info {
    font-size: 0.9rem;
    color: #8b949e;
    margin-top: 0.5rem;
}
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000/api"

# ==============================================================================
# API FETCHERS
# ==============================================================================
@st.cache_data(ttl=1)
def fetch_tickers():
    try:
        res = requests.get(f"{API_BASE_URL}/tickers", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        return {}

def fetch_signal(ticker, refresh=False):
    try:
        url = f"{API_BASE_URL}/signal/{ticker}"
        if refresh: 
            url += "?refresh=true"
        res = requests.get(url, timeout=45) # Long timeout in case model needs loading
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.error(f"Failed to fetch signal for {ticker}: {str(e)}")
    return None

def fetch_market_data(ticker):
    try:
        res = requests.get(f"{API_BASE_URL}/market/{ticker}", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        pass
    return None

def fetch_whatif(ticker, price_shock, volume_shock, vix_shock):
    try:
        payload = {
            "price_shock": price_shock,
            "volume_shock": volume_shock,
            "vix_shock": vix_shock
        }
        res = requests.post(f"{API_BASE_URL}/whatif/{ticker}", json=payload, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

# ==============================================================================
# SIDEBAR
# ==============================================================================
_LOGO = Path(__file__).parent / "dashboard_static" / "soligence_logo.png"
if _LOGO.exists():
    st.sidebar.image(str(_LOGO), width=60)
else:
    st.sidebar.markdown("## 📈")
st.sidebar.title("SOLiGence")
st.sidebar.subheader("Ensemble AI Barometer")
st.sidebar.markdown("---")

system_status = fetch_tickers()
if not system_status:
    st.error("Backend API is unreachable. Is `dashboard_app.py` running?")
    st.stop()

valid_tickers = [t for t in system_status.keys()]
selected_ticker = st.sidebar.selectbox("Select Champion Asset", valid_tickers)

# Status pill
t_status = system_status.get(selected_ticker, {})
if t_status.get("loaded"):
    st.sidebar.success("🟢 Model Active")
elif t_status.get("loading"):
    st.sidebar.warning("⏳ Model Loading...")
elif t_status.get("model_exists"):
    st.sidebar.info("⚪ Model on Disk (Idle)")
else:
    st.sidebar.error("🔴 Model Missing")

refresh_data = st.sidebar.button("🔄 Refresh Live Market Data", use_container_width=True)

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================
if not t_status.get("model_exists"):
    st.warning(f"No trained artifacts found for {selected_ticker}. Please run barometer_core.py training sequence.")
    st.stop()

with st.spinner(f"Synchronizing ensemble models for {selected_ticker}..."):
    sig_data = fetch_signal(selected_ticker, refresh=refresh_data)

if sig_data is None:
    st.error("Could not load signal data. Backend may be still initializing...")
    st.stop()

st.title(f"📈 {selected_ticker} Advanced Forecasting")
st.markdown(f"**Last Close:** ${sig_data['last_close']} | **Date:** {sig_data['last_date']}")
st.markdown("---")

# 1. SIGNAL METRICS GRID
horizons = [("t1", "T+1 Day"), ("t5", "T+5 Days"), ("t21", "T+21 Days"), ("t63", "T+63 Days")]
cols = st.columns(len(horizons))

for idx, (h_key, h_label) in enumerate(horizons):
    if h_key in sig_data["horizons"]:
        v = sig_data["horizons"][h_key]
        sig_color_cls = f"signal-{v['signal']}"
        arrow = "🔺" if v["pct_change"] >= 0 else "🔻"
        pct_color = "#3fb950" if v["pct_change"] >= 0 else "#f85149"
        
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="signal-title">{h_label} Horizon</div>
                <div class="signal-value {sig_color_cls}">{v['signal']} {v['emoji']}</div>
                <div style="font-size: 1.3rem; margin-top: 10px;">
                    Target: <b>${v['price_pred']:.2f}</b> 
                    <span style="color: {pct_color}; font-size: 1.1rem;">({arrow} {v['pct_change']}%)</span>
                </div>
                <div class="meta-info">
                    Confidence: <b>{v['confidence']}%</b> &nbsp;|&nbsp; Up-Prob: <b>{v['up_prob']}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# 2. MARKET DATA VISUALIZATION
market = fetch_market_data(selected_ticker)
if market:
    st.subheader("Market Technical Profile")
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Scatter(x=market['dates'], y=market['close'], mode='lines', name='Close Price', line=dict(color='#58a6ff', width=2)))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=market['dates'], y=market['bb_upper'], line=dict(color='rgba(255,255,255,0)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=market['dates'], y=market['bb_lower'], fill='tonexty', fillcolor='rgba(139, 148, 158, 0.1)', line=dict(color='rgba(255,255,255,0)', width=0), showlegend=False, name='Bollinger Band'))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="Price ($)",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


# 3. WHAT-IF STRESS TESTING
st.markdown("---")
st.subheader("⚡ Stress Testing (What-If Scenario)")

with st.container():
    c1, c2, c3 = st.columns(3)
    p_shock = c1.slider("Price Shock (%)", min_value=-20.0, max_value=20.0, value=-5.0, step=0.5) / 100.0
    v_shock = c2.slider("Volume Shock (%)", min_value=-50.0, max_value=200.0, value=50.0, step=5.0) / 100.0
    vix_shock= c3.slider("VIX (Fear) Shock (pts)", min_value=-10.0, max_value=40.0, value=15.0, step=1.0)
    
    if st.button("Run Portfolio Stress Test", use_container_width=True):
        with st.spinner("Calculating deep-ensemble shocks..."):
            whatif_res = fetch_whatif(selected_ticker, p_shock, v_shock, vix_shock)
            if whatif_res and "results" in whatif_res:
                st.success(f"Scenario Applied: {whatif_res['scenario']}")
                
                # Render table
                table_data = []
                for h in ["t1", "t5", "t21", "t63"]:
                    if h in whatif_res["results"]:
                        rd = whatif_res["results"][h]
                        table_data.append({
                            "Horizon": h.upper(),
                            "Base Target ($)": rd['base_price'],
                            "Shocked Target ($)": rd['shocked_price'],
                            "Impact ($)": rd['delta'],
                            "Deviation (%)": rd['delta_pct'],
                            "Defense Posture": rd['impact']
                        })
                
                if table_data:
                    df_wi = pd.DataFrame(table_data)
                    st.dataframe(
                        df_wi.style.map(lambda x: "color: #3fb950;" if x == "PROTECTIVE" else "color: #f85149;" if x == "FOLLOWING" else "", subset=["Defense Posture"]),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.error("Stress test failed.")
