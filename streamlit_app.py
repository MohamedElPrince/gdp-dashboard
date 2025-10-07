import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="D86 Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #fed7aa;
        color: #9a3412;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 D86 Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Automated Daily Prediction Reports with Store-Level Intelligence**")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Date selector
    prediction_date = st.date_input(
        "Prediction Date",
        value=datetime.now() + timedelta(days=1),
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=7)
    )
    
    # Store filter
    store_options = ['All Stores'] + [f'Store {i}' for i in [101, 112, 114, 120, 312, 401, 433, 478, 542, 2289, 2355, 2450, 2580, 2588, 2738, 2841, 5481, 5496, 5587, 5676, 5687, 5721]]
    selected_store = st.selectbox("Select Store", store_options)
    
    # Risk threshold
    risk_threshold = st.slider("Risk Score Threshold", 0, 100, 60, 5)
    
    # Category filter
    category_options = ['All Categories', 'SWEET FOOD', 'SAVORY FOOD', 'BAKERY', 'BEVERAGE']
    selected_category = st.selectbox("Category Filter", category_options)
    
    st.markdown("---")
    st.markdown("### 📈 Model Info")
    st.metric("Model Accuracy", "85.3%")
    st.metric("Last Updated", "2025-10-06")
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Options")
    if st.button("📊 Export to Excel", use_container_width=True):
        st.success("Export functionality ready!")
    if st.button("📧 Email Report", use_container_width=True):
        st.success("Email sent to stakeholders!")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Generate synthetic prediction data (replace with actual Incorta data)
@st.cache_data
def generate_prediction_data():
    np.random.seed(42)
    stores = [101, 112, 114, 120, 312, 401, 433, 478, 542, 2289, 2355, 2450, 2580, 2588, 2738, 2841, 5481, 5496, 5587, 5676, 5687, 5721]
    
    items = [
        {'name': 'BLUEBERRY MUFFIN', 'num': '011158027', 'cat': 'SWEET FOOD'},
        {'name': 'BACON GOUDA SANDWICH', 'num': '011096091', 'cat': 'SAVORY FOOD'},
        {'name': 'CHOCOLATE CHIP COOKIE', 'num': '011110086', 'cat': 'SWEET FOOD'},
        {'name': 'RACCOON CAKE POP', 'num': '011160880', 'cat': 'SWEET FOOD'},
        {'name': 'BUTTER CROISSANT', 'num': '011083122', 'cat': 'SWEET FOOD'},
        {'name': 'LEMON LOAF', 'num': '011079227', 'cat': 'SWEET FOOD'},
        {'name': 'HAM CHEESE CROISSANT', 'num': '011130008', 'cat': 'SAVORY FOOD'},
        {'name': 'SPINACH FETA WRAP', 'num': '011104540', 'cat': 'SAVORY FOOD'},
        {'name': 'CHOCOLATE CROISSANT', 'num': '011083123', 'cat': 'SWEET FOOD'},
        {'name': 'TURKEY BACON SANDWICH', 'num': '011041854', 'cat': 'SAVORY FOOD'},
        {'name': 'PUMPKIN LOAF', 'num': '011162513', 'cat': 'SWEET FOOD'},
        {'name': 'BIRTHDAY CAKE POP', 'num': '011037822', 'cat': 'SWEET FOOD'},
        {'name': 'PLAIN BAGEL', 'num': '011167575', 'cat': 'BAKERY'},
        {'name': 'CHEESE DANISH', 'num': '011083579', 'cat': 'SWEET FOOD'},
        {'name': 'BACON SAUSAGE WRAP', 'num': '011112214', 'cat': 'SAVORY FOOD'},
    ]
    
    data = []
    for store in stores:
        for item in items[:12]:  # Use subset of items
            on_hand = np.random.randint(0, 150)
            forecast = np.random.randint(5, 80)
            allocation = np.random.choice([0, 0, np.random.randint(10, 50)], p=[0.3, 0.2, 0.5])
            neighbor_avg = np.random.randint(20, 100)
            
            # Calculate risk score based on features
            inventory_risk = max(0, (forecast - on_hand) / max(forecast, 1)) * 40
            allocation_risk = 20 if allocation == 0 and on_hand < forecast else 0
            neighbor_risk = 15 if neighbor_avg > on_hand * 1.5 else 0
            historical_risk = np.random.randint(0, 25)
            
            risk_score = min(100, int(inventory_risk + allocation_risk + neighbor_risk + historical_risk))
            
            # Determine reason
            if on_hand < forecast and allocation == 0:
                reason = "Critical: Low inventory + No allocation"
            elif on_hand < forecast * 0.5:
                reason = "High: Severely understocked"
            elif allocation == 0:
                reason = "Medium: No orders in flight"
            else:
                reason = "Low: Adequate supply chain"
            
            data.append({
                'store_number': store,
                'item_name': item['name'],
                'item_number': item['num'],
                'category': item['cat'],
                'risk_score': risk_score,
                'on_hand': on_hand,
                'forecast_demand': forecast,
                'allocated_qty': allocation,
                'neighbor_avg': neighbor_avg,
                'gap': forecast - on_hand,
                'reason': reason,
                'prediction_date': prediction_date.strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(data)

# Load data
df = generate_prediction_data()

# Apply filters
if selected_store != 'All Stores':
    store_num = int(selected_store.split()[1])
    df = df[df['store_number'] == store_num]

if selected_category != 'All Categories':
    df = df[df['category'] == selected_category]

# Filter by risk threshold
high_risk_df = df[df['risk_score'] >= risk_threshold].sort_values('risk_score', ascending=False)

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🚨 Critical Items",
        len(df[df['risk_score'] >= 80]),
        delta=f"{len(df[df['risk_score'] >= 80]) - 15}",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "⚠️ High Risk Items",
        len(df[(df['risk_score'] >= 60) & (df['risk_score'] < 80)]),
        delta=f"{len(df[(df['risk_score'] >= 60) & (df['risk_score'] < 80)]) - 28}"
    )

with col3:
    st.metric(
        "📦 Total Items Monitored",
        len(df),
        delta="Active"
    )

with col4:
    st.metric(
        "🏪 Stores Analyzed",
        df['store_number'].nunique(),
        delta="Real-time"
    )

with col5:
    avg_risk = df['risk_score'].mean()
    st.metric(
        "📊 Avg Risk Score",
        f"{avg_risk:.1f}",
        delta=f"{avg_risk - 45:.1f}"
    )

st.markdown("---")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 High Risk Predictions", 
    "📊 Risk Distribution", 
    "🏪 Store Analysis",
    "📈 Trends & Patterns",
    "💡 Recommendations"
])

with tab1:
    st.subheader(f"🚨 High Risk Items (Score ≥ {risk_threshold}) - {prediction_date.strftime('%B %d, %Y')}")
    
    if len(high_risk_df) > 0:
        st.info(f"**{len(high_risk_df)} items** are predicted to have high D86 risk tomorrow. Take action now to prevent stockouts!")
        
        # Add risk category column for display
        def get_risk_category(score):
            if score >= 80:
                return "🔴 Critical"
            elif score >= 60:
                return "🟠 High"
            elif score >= 40:
                return "🟡 Medium"
            else:
                return "🟢 Low"
        
        high_risk_df['risk_category'] = high_risk_df['risk_score'].apply(get_risk_category)
        
        # Display table
        display_df = high_risk_df[['store_number', 'item_name', 'item_number', 'category', 
                                     'risk_category', 'risk_score', 'on_hand', 'forecast_demand', 
                                     'allocated_qty', 'neighbor_avg', 'gap', 'reason']].copy()
        
        display_df.columns = ['Store', 'Item Name', 'Item #', 'Category', 'Risk Level', 
                               'Risk Score', 'On Hand', 'Forecast', 'Allocated', 
                               'Neighbor Avg', 'Gap', 'Primary Reason']
        
        # Style the dataframe
        def highlight_risk(row):
            if row['Risk Score'] >= 80:
                return ['background-color: #fee2e2'] * len(row)
            elif row['Risk Score'] >= 60:
                return ['background-color: #fed7aa'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download High Risk Items (CSV)",
            data=csv,
            file_name=f"d86_predictions_{prediction_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No high-risk items found for the selected criteria!")

with tab2:
    st.subheader("📊 Risk Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_bins = pd.cut(df['risk_score'], bins=[0, 40, 60, 80, 100], 
                           labels=['Low (0-39)', 'Medium (40-59)', 'High (60-79)', 'Critical (80-100)'])
        risk_counts = risk_bins.value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=['#10b981', '#eab308', '#f59e0b', '#ef4444']
        )])
        fig_pie.update_layout(title="Risk Level Distribution", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk by category
        category_risk = df.groupby('category')['risk_score'].mean().sort_values(ascending=False)
        fig_bar = px.bar(
            x=category_risk.index,
            y=category_risk.values,
            title="Average Risk Score by Category",
            labels={'x': 'Category', 'y': 'Avg Risk Score'},
            color=category_risk.values,
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk score histogram
    fig_hist = px.histogram(
        df,
        x='risk_score',
        nbins=20,
        title="Distribution of Risk Scores Across All Items",
        labels={'risk_score': 'Risk Score', 'count': 'Number of Items'},
        color_discrete_sequence=['#3b82f6']
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("🏪 Store-Level Analysis")
    
    # Store risk summary
    store_summary = df.groupby('store_number').agg({
        'risk_score': 'mean',
        'item_name': 'count',
        'on_hand': 'sum',
        'forecast_demand': 'sum'
    }).reset_index()
    store_summary.columns = ['Store', 'Avg Risk Score', 'Items Monitored', 'Total On Hand', 'Total Forecast']
    store_summary['Critical Items'] = df[df['risk_score'] >= 80].groupby('store_number').size().reindex(store_summary['Store'], fill_value=0).values
    
    store_summary = store_summary.sort_values('Avg Risk Score', ascending=False)
    
    # Top 10 highest risk stores
    st.markdown("#### 🎯 Top 10 Highest Risk Stores")
    top_stores = store_summary.head(10)
    
    fig_stores = go.Figure()
    fig_stores.add_trace(go.Bar(
        x=top_stores['Store'],
        y=top_stores['Avg Risk Score'],
        marker_color='#ef4444',
        name='Avg Risk Score'
    ))
    fig_stores.update_layout(
        title="Stores with Highest Average Risk Scores",
        xaxis_title="Store Number",
        yaxis_title="Average Risk Score",
        height=400
    )
    st.plotly_chart(fig_stores, use_container_width=True)
    
    # Store details table
    st.markdown("#### 📋 Complete Store Summary")
    st.dataframe(
        store_summary.style.background_gradient(subset=['Avg Risk Score'], cmap='Reds'),
        use_container_width=True,
        height=400
    )
    
    # Individual store deep dive
    st.markdown("---")
    st.markdown("#### 🔍 Individual Store Deep Dive")
    store_select = st.selectbox(
        "Select a store for detailed analysis:",
        sorted(df['store_number'].unique())
    )
    
    store_data = df[df['store_number'] == store_select].sort_values('risk_score', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Items at Risk (≥60)", len(store_data[store_data['risk_score'] >= 60]))
    with col2:
        st.metric("Avg Risk Score", f"{store_data['risk_score'].mean():.1f}")
    with col3:
        st.metric("Total Inventory", f"{store_data['on_hand'].sum():,}")
    with col4:
        st.metric("Forecast Gap", f"{store_data['gap'].sum():,}")
    
    # Store items table
    st.dataframe(
        store_data[['item_name', 'category', 'risk_score', 'on_hand', 'forecast_demand', 'allocated_qty', 'reason']],
        use_container_width=True,
        height=400
    )

with tab4:
    st.subheader("📈 Trends & Patterns")
    
    # Simulated historical trend data
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    historical_data = pd.DataFrame({
        'Date': dates,
        'D86_Rate': [12.5, 13.8, 14.2, 15.1, 13.9, 12.7, 13.3],
        'Critical_Items': [10, 13, 15, 18, 14, 11, 12],
        'Avg_Risk_Score': [42, 45, 48, 52, 47, 44, 46]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # D86 rate trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['D86_Rate'],
            mode='lines+markers',
            name='D86 Rate (%)',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        ))
        fig_trend.update_layout(
            title="Historical D86 Rate Trend (Last 7 Days)",
            xaxis_title="Date",
            yaxis_title="D86 Rate (%)",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Critical items trend
        fig_critical = go.Figure()
        fig_critical.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Critical_Items'],
            mode='lines+markers',
            name='Critical Items',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=8)
        ))
        fig_critical.update_layout(
            title="Critical Items Trend (Score ≥ 80)",
            xaxis_title="Date",
            yaxis_title="Number of Items",
            height=400
        )
        st.plotly_chart(fig_critical, use_container_width=True)
    
    # Pattern insights
    st.markdown("#### 🔍 Pattern Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📅 Day of Week Patterns**")
        st.info("Mondays and Fridays show 23% higher D86 rates due to weekend demand fluctuations.")
    
    with col2:
        st.markdown("**🍰 Category Patterns**")
        st.warning("Sweet Food items have 18% higher D86 rates than Savory items.")
    
    with col3:
        st.markdown("**⏰ Time Patterns**")
        st.error("Peak D86 occurrence: 10-11 AM (breakfast rush) and 2-3 PM (lunch rush).")

with tab5:
    st.subheader("💡 Actionable Recommendations")
    
    st.markdown("""
    ### 🎯 Immediate Actions Required
    
    Based on the D86 predictions, here are prioritized recommendations:
    """)
    
    # Generate recommendations based on data
    critical_items = df[df['risk_score'] >= 80].sort_values('risk_score', ascending=False)
    
    if len(critical_items) > 0:
        st.markdown("#### 🚨 Critical Priority (Next 4 Hours)")
        
        for idx, row in critical_items.head(5).iterrows():
            with st.expander(f"Store {row['store_number']} - {row['item_name']} (Risk: {row['risk_score']})"):
                st.markdown(f"""
                **Current Situation:**
                - On Hand: {row['on_hand']} units
                - Forecasted Demand: {row['forecast_demand']} units
                - Gap: {row['gap']} units
                - Allocated: {row['allocated_qty']} units
                
                **Recommended Actions:**
                1. **Immediate**: Place emergency order for {max(row['gap'], 20)} units
                2. **Transfer**: Check stores {row['store_number']-1}, {row['store_number']+1} for available inventory (avg: {row['neighbor_avg']} units)
                3. **Substitute**: Offer similar items if stockout occurs
                4. **Alert**: Notify store manager and regional supervisor
                
                **Expected Impact:** Prevents estimated ${row['forecast_demand'] * 5:.2f} in lost sales
                """)
    
    st.markdown("---")
    
    # Strategic recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 Supply Chain Optimization")
        st.markdown("""
        - **Increase PAR Levels** for top 10 high-risk items by 15%
        - **Expedite Orders** for items with 0 allocation
        - **Adjust Delivery Schedule** for high-volume stores
        - **Build Safety Stock** for items with chronic D86 patterns
        """)
    
    with col2:
        st.markdown("#### 🔄 Operational Improvements")
        st.markdown("""
        - **Inter-Store Transfers**: Implement automated transfer system
        - **Real-Time Monitoring**: Set up alerts for risk score > 70
        - **Demand Sensing**: Improve forecast accuracy with ML
        - **Vendor Management**: Negotiate faster lead times
        """)
    
    # Cost impact analysis
    st.markdown("---")
    st.markdown("#### 💰 Financial Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    estimated_lost_sales = len(critical_items) * 25 * 5  # items * avg demand * avg price
    prevention_cost = len(critical_items) * 50  # emergency order cost per item
    net_benefit = estimated_lost_sales - prevention_cost
    
    with col1:
        st.metric("Potential Lost Sales", f"${estimated_lost_sales:,.2f}", delta="Risk")
    with col2:
        st.metric("Prevention Cost", f"${prevention_cost:,.2f}", delta="Investment")
    with col3:
        st.metric("Net Benefit", f"${net_benefit:,.2f}", delta=f"+{(net_benefit/estimated_lost_sales)*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><strong>D86 Prediction System v2.0</strong> | Powered by ML & Incorta Data Platform</p>
    <p>Last Updated: {dt_now} | Data Refresh: Every 15 minutes</p>
    <p>For support, contact: supply-chain-analytics@company.com</p>
</div>
""".format(dt_now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)