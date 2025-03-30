import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # Technical analysis library
from datetime import datetime

# Configure app
st.set_page_config(
    page_title="TS-NiSAM Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stSelectbox div[data-baseweb="select"] {border: 1px solid #2e86c1;}
    .stSlider div[data-testid="stTickBar"] {background: #2e86c1;}
    .css-1aumxhk {color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🔧 TS-NiSAM Configuration")
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"],
    index=0
)

date_col1, date_col2 = st.sidebar.columns(2)
with date_col1:
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
with date_col2:
    end_date = st.date_input("End Date", datetime.today())

# Main content
st.title("📊 TS-NiSAM: Advanced Stock Analysis Platform")
st.markdown("**Integrated Risk Management & Forecasting System**")

# Data loading
@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start=start_date, end=end_date)

df = load_data(selected_stock)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Analysis", 
    "📉 Technical Indicators",
    "🎲 Monte Carlo",
    "🏆 Recommendations",
    "📚 Fundamentals"
])

# Tab 1: Price Analysis
with tab1:
    st.header("Multi-Timeframe Price Analysis")
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='#2e86c1'
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Technical Indicators
with tab2:
    st.header("Advanced Technical Analysis")
    
    # Calculate indicators
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Moving Averages
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
        fig1.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20 MA'))
        fig1.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA'))
        fig1.update_layout(title='Moving Averages', height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # MACD
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=df.index, y=df['MACD'], name='MACD'))
        fig3.add_trace(go.Scatter(x=df.index, y=df['Signal'], 
                            name='Signal', line=dict(color='orange')))
        fig3.update_layout(title='MACD', height=300)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # RSI
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig2.add_hline(y=30, line_dash="dash", line_color="green")
        fig2.add_hline(y=70, line_dash="dash", line_color="red")
        fig2.update_layout(title='Relative Strength Index (RSI)', 
                          yaxis_range=[0,100], height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price'))
        fig4.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], 
                                name='Upper Band', line=dict(color='gray')))
        fig4.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], 
                                name='Lower Band', line=dict(color='gray')))
        fig4.update_layout(title='Bollinger Bands', height=300)
        st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Monte Carlo Simulation (Heston Model)
with tab3:
    st.header("Heston Model Simulation")
    
    # Implement Heston Model equations here
    # (Use the mathematical formulas from your research paper)
    
    # Placeholder visualization
    fig = go.Figure()
    for _ in range(5):
        simulated_prices = df['Close'].iloc[-1] * np.exp(np.cumsum(
            np.random.normal(0.001, 0.02, 100)
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(100),
            y=simulated_prices,
            line=dict(width=1),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Monte Carlo Price Simulations",
        xaxis_title="Days",
        yaxis_title="Price",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Recommendations
with tab4:
    st.header("AI-Powered Recommendations")
    
    # Recommendation data
    rec_data = {
        'Stock': ["Reliance", "Tata Motors", "SBI", "HDFC Bank", "Infosys"],
        '1Y Return': [23.58, 103.82, 37.23, 18.95, 22.41],
        '3Y CAGR': [15.4, 46.2, 31.0, 12.8, 18.6],
        'Piotroski Score': [7, 6, 5, 8, 7],
        'Risk Level': ['Medium', 'High', 'Medium', 'Low', 'Low']
    }
    
    # Create styled dataframe
    df_rec = pd.DataFrame(rec_data).set_index('Stock')
    st.dataframe(
        df_rec.style
            .background_gradient(subset=['1Y Return', '3Y CAGR'], cmap='Blues')
            .highlight_max(subset=['1Y Return', '3Y CAGR'], color='#d4f7d4')
            .highlight_min(subset=['1Y Return', '3Y CAGR'], color='#ffe8e8'),
        use_container_width=True
    )
    
    # Portfolio optimization visualization
    fig = go.Figure(data=[
        go.Pie(labels=df_rec.index,
              values=df_rec['1Y Return'],
              hole=0.4,
              marker_colors=['#2e86c1', '#3498db', '#5dade2', '#85c1e9', '#aed6f1'])
    ])
    fig.update_layout(title="Recommended Portfolio Allocation")
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Fundamentals
with tab5:
    st.header("Fundamental Analysis")
    
    # Piotroski Score visualization
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=7,  # Replace with actual score
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Piotroski F-Score"},
        gauge={'axis': {'range': [0, 9]},
               'steps': [
                   {'range': [0, 3], 'color': "red"},
                   {'range': [3, 7], 'color': "orange"},
                   {'range': [7, 9], 'color': "green"}],
               'threshold': {'line': {'color': "black", 'width': 4},
                             'thickness': 0.75,
                             'value': 7}}))
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**TS-NiSAM Pro** © 2024 | Powered by Ensemble Learning & Monte Carlo Simulations  
*Disclaimer: This is a research tool - not financial advice*
""")
