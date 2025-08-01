import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Car Sales Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    /* Increase chart container width */
    .element-container {
        width: 100%;
    }
    /* Custom styling for navigation boxes */
    .nav-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .nav-box:hover {
        background-color: #e0e2e6;
    }
    .nav-box.active {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    /* Handle scrolling to top immediately when page changes */
    html {
        scroll-behavior: auto !important;
        scroll-padding-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('Car_Sales.csv')
    data.dropna(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    return data

data = load_data()

st.markdown("""
<script>
    // More aggressive scroll to top function
    function forceScrollToTop() {
        window.scrollTo({
            top: 0,
            left: 0,
            behavior: 'auto'  // Use 'auto' instead of 'smooth' for immediate scroll
        });
        
        // Apply additional force scroll with setTimeout
        setTimeout(function() {
            window.scrollTo(0, 0);
            document.body.scrollTop = 0;
            document.documentElement.scrollTop = 0;
        }, 50);
    }
    
    // Execute on page load
    forceScrollToTop();
    
    // Add click event listeners to all navigation buttons
    document.addEventListener('DOMContentLoaded', function() {
        const buttons = document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stButton"] button');
        buttons.forEach(button => {
            button.addEventListener('click', function() {
                // Force scroll on button click
                forceScrollToTop();
            });
        });
    });
    
    // MutationObserver as backup
    const targetNode = document.querySelector('body');
    const config = { childList: true, subtree: true };
    const callback = function(mutationsList, observer) {
        forceScrollToTop();
    };
    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);
</script>
""", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='margin-bottom:0px;'>Functions</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin-top:0px; margin-bottom:10px;'>", unsafe_allow_html=True)

pages = {
    "📊 Dashboard Overview": "#4e79a7",  # Blue 
    "👥 Customer Segmentation": "#f28e2c",  # Orange
    "🏢 Dealer Performance": "#e15759",  # Red
    "📈 Cohort Analysis": "#76b7b2",  # Teal
    "💰 Product Profitability": "#59a14f",  # Green
    "🎯 Strategic Recommendations": "#af7aa1"  # Purple
}

if 'current_page' not in st.session_state:
    st.session_state.current_page = "📊 Dashboard Overview"

# Generate navigation boxes using Streamlit components
for i, (page_name, color) in enumerate(pages.items()):
    is_active = st.session_state.current_page == page_name
    
    if st.sidebar.button(
        page_name, 
        key=f"btn_{page_name}",
        on_click=lambda pn=page_name: st.session_state.update({"current_page": pn}),
        type="primary" if is_active else "secondary",
        use_container_width=True
    ):
        pass
    
    st.markdown(f"""
    <style>
        /* Target the specific button by its key */
        [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="{0 if is_active else 1}"][data-testid="{f"btn_{page_name}"}"] {{
            background-color: {color} !important;
            opacity: {1 if is_active else 0.7};
            font-weight: {700 if is_active else 400};
            color: white !important;
            border: none !important;
        }}
        
        /* Hover effect for the button */
        [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="{0 if is_active else 1}"][data-testid="{f"btn_{page_name}"}"]:hover {{
            opacity: 1 !important;
            background-color: {color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin-top:5px; margin-bottom:5px;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="font-size:0.85em;">
This dashboard provides comprehensive insights into car sales data, customer segmentation, and dealer performance metrics.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        padding-top: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# Function to display the Dashboard Overview
def dashboard_overview():
    st.markdown("<div class='main-header'>Car Sales Analytics Dashboard</div>", unsafe_allow_html=True)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("👥 **Customer Segmentation**\nExplore customer demographics and purchasing behavior")
    with col2:
        st.info("🏢 **Dealer Performance**\nAnalyze dealer sales performance and regional insights")
    with col3:
        st.info("📈 **Cohort Analysis**\nTrack car sales volume and price elasticity over time")
    
    col4, col5 = st.columns(2)
    with col4:
        st.info("💰 **Product Profitability**\nView car models revenue and sales volume metrics")
    with col5:
        st.info("🎯 **Strategic Recommendations**\nGet insights for product and dealer partnerships")
    
    # summary statistics
    st.markdown("<div class='section-header'>Quick Summary Statistics</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Total Sales
        with col1:
            total_sales_value = data['Price ($)'].sum()
            if total_sales_value >= 1_000_000:
                formatted_sales = f"${total_sales_value/1_000_000:.1f}M"
            else:
                formatted_sales = f"${total_sales_value:,.2f}"
            st.metric("Total Sales", formatted_sales)
        
        # Total Customers
        with col2:
            if 'Customer Name' in data.columns:
                total_customers = f"{len(data['Customer Name'].unique()):,}"
            else:
                customer_cols = [col for col in data.columns if 'customer' in col.lower()]
                if customer_cols:
                    total_customers = f"{len(data[customer_cols[0]].unique()):,}"
                else:
                    total_customers = "N/A"
            st.metric("Total Customers", total_customers)
        
        # Average Car Price
        with col3:
            avg_car_price = f"${data['Price ($)'].mean():,.2f}"
            st.metric("Average Car Price", avg_car_price)
        
        # Total Purchases
        with col4:
            total_purchases = f"{len(data):,}"
            st.metric("Total Purchases", total_purchases)
            
    except Exception as e:
        st.error(f"Error calculating summary statistics: {e}")
        st.write("Available columns in dataset:", data.columns.tolist())



# TASK1 Customer Segmentation
def customer_segmentation():
    st.markdown("<div class='main-header'>Customer Segmentation and Insights</div>", unsafe_allow_html=True)

    # K-means Clustering with consistent segment assignments
    X = data[['Annual Income', 'Price ($)']]
    kmeans = KMeans(n_clusters=4, random_state=42)  # Set random_state for reproducibility
    data['Segment'] = kmeans.fit_predict(X)

    # Get the cluster centers and map them to expected segments
    centers = kmeans.cluster_centers_
    # Sort centers by income (first feature) to get consistent ordering
    center_income = centers[:, 0]
    # Get the indices that would sort the centers by income
    sorted_indices = np.argsort(center_income)
    # Create a mapping from original cluster ID to ordered ID
    cluster_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    # Apply the mapping to get consistent segment labels
    data['Segment'] = data['Segment'].map(cluster_mapping)

   
    segment_display_names = {
        0: "Low",
        1: "Middle",
        2: "Upper-Middle",
        3: "High"
    }

    # Summary of Customer Segments 
    st.subheader("Customer Segments Summary")
    summary = data.groupby('Segment').agg({
        'Customer Name': 'count',
        'Annual Income': 'mean',
        'Price ($)': 'mean'
    }).reset_index()


    summary['Segment'] = summary['Segment'].map(segment_display_names)

    summary = summary.rename(columns={'Customer Name': 'Customer Count'})

    st.dataframe(summary)


    st.subheader("Insights:")
    total_customers = len(data['Customer Name'].unique())  
    avg_annual_income = data['Annual Income'].mean()
    avg_car_price = data['Price ($)'].mean()
    total_purchases = len(data)  # Total number of rows/purchases


    if 'Color' in data.columns:
        most_common_color = data['Color'].value_counts().idxmax()

    if 'Model' in data.columns:
        most_popular_model = data['Model'].value_counts().idxmax()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"• Total Customers: {total_customers}")
        st.markdown(f"• Average Annual Income: ${avg_annual_income:,.2f}")
        st.markdown(f"• Average Car Price: ${avg_car_price:,.2f}")
    with col2:
        st.markdown(f"• Total Purchases: {total_purchases}")
        st.markdown(f"• Most Common Car Color: {most_common_color}")
        st.markdown(f"• Most Popular Model: {most_popular_model}")

    # Gender Distribution
    st.subheader("Gender Distribution of Customers")
    gender_counts = data['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']

    gender_fig = px.bar(gender_counts, x='Gender', y='Count', color='Gender',
                    color_discrete_map={'Male': 'blue', 'Female': 'pink'},
                    title='Gender Distribution of Customers')
    gender_fig.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(gender_fig, use_container_width=True)

    # Average Price by Customer Segment 
    st.subheader("Average Price by Customer Segment")
    
    avg_price = data.groupby('Segment')['Price ($)'].mean().reset_index()
    price_fig = px.bar(avg_price, x='Segment', y='Price ($)',
                    title='Average Price by Customer Segment',
                    color='Segment', color_continuous_scale='blues')
    price_fig.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600, 
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # Annual Income Distribution 
    st.subheader("Annual Income Distribution")
    income_fig = px.histogram(data, x='Annual Income',
                        title='Annual Income Distribution',
                        nbins=30,
                        color_discrete_sequence=['blue'])
    income_fig.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(income_fig, use_container_width=True)

    # Purchase Trends Over Time 
    st.subheader("Purchase Trends Over Time")
    data['Month_Period'] = data['Date'].dt.to_period('M')
    purchase_trends = data.groupby('Month_Period').size().reset_index(name='Number of Purchases')
    purchase_trends['Month'] = purchase_trends['Month_Period'].dt.to_timestamp()
    trend_fig = px.line(purchase_trends, x='Month', y='Number of Purchases',
                    title='Purchase Trends Over Time')
    trend_fig.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    # Car Type Preferences by Gender 
    st.subheader("Car Type Preferences by Gender")
    
    car_gender_counts = data.groupby(['Body Style', 'Gender']).size().reset_index(name='Count')
    car_gender_fig = px.bar(car_gender_counts, x='Body Style', y='Count', color='Gender',
                        color_discrete_map={'Male': 'lightblue', 'Female': 'blue'},
                        barmode='group', title='Car Type Preferences by Gender')
    car_gender_fig.update_layout(
        xaxis_title='Car Type',
        yaxis_title='Count',
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(car_gender_fig, use_container_width=True)

    # Car Type Preference by Income Segment
    st.subheader("Car Type Preference by Income Segment")
    # Create income segments as before
    data['Income Segment'] = pd.qcut(data['Annual Income'], 4, labels=['Low', 'Middle', 'Upper-Middle', 'High'])
    car_income_counts = data.groupby(['Body Style', 'Income Segment'], observed=False).size().reset_index(name='Count')
    car_income_fig = px.bar(car_income_counts, x='Body Style', y='Count', color='Income Segment',
                        barmode='group', title='Car Type Preference by Income Segment',
                        color_discrete_sequence=['lightblue', 'royalblue', 'navy', 'darkblue'])
    car_income_fig.update_layout(
        xaxis_title='Car Type',
        yaxis_title='Count',
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(car_income_fig, use_container_width=True)

# TASK2 Dealer Performance
def dealer_performance():
    st.markdown("<div class='main-header'>Dealer Performance Dashboard</div>", unsafe_allow_html=True)

    # Performance Metrics Over Time
    monthly_performance = data.groupby(['Year', 'Month', 'Dealer_Name']).agg(
        Total_Sales=('Car_id', 'count'),
        Total_Revenue=('Price ($)', 'sum')
    ).reset_index()

    monthly_performance['Date'] = pd.to_datetime(monthly_performance[['Year', 'Month']].assign(DAY=1))

    # Total sales and revenue for each dealer
    dealer_performance = monthly_performance.groupby('Dealer_Name').agg(
        Total_Sales=('Total_Sales', 'sum'),
        Total_Revenue=('Total_Revenue', 'sum')
    ).reset_index()

    # Rank dealers based on total sales
    dealer_performance['Sales_Rank'] = dealer_performance['Total_Sales'].rank(ascending=False)
    dealer_performance['Revenue_Rank'] = dealer_performance['Total_Revenue'].rank(ascending=False)

    top_dealers = dealer_performance.sort_values(by='Total_Sales', ascending=False)

    # Regional Analysis
    regional_performance = data.groupby('Dealer_Region').agg(
        Total_Sales=('Car_id', 'count'),
        Total_Revenue=('Price ($)', 'sum')
    ).reset_index()


    st.subheader('Top-Performing Dealers')
    st.dataframe(top_dealers[['Dealer_Name', 'Total_Sales', 'Total_Revenue', 'Sales_Rank', 'Revenue_Rank']].reset_index(drop=True))

    # Dealer Selection
    dealer_selection = st.selectbox("Select Dealer", monthly_performance['Dealer_Name'].unique())

    filtered_data = monthly_performance[monthly_performance['Dealer_Name'] == dealer_selection]

    # Total Sales for selected dealer
    st.subheader(f'Total Sales for {dealer_selection} Over Time')
    fig_selected_sales = px.line(
        filtered_data,
        x='Date',
        y='Total_Sales',
        title=f'Total Sales Over Time for {dealer_selection}',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_selected_sales.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(fig_selected_sales, use_container_width=True)
    
    # Total Revenue for selected dealer
    st.subheader(f'Total Revenue for {dealer_selection} Over Time')
    fig_selected_revenue = px.line(
        filtered_data,
        x='Date',
        y='Total_Revenue',
        title=f'Total Revenue Over Time for {dealer_selection}',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_selected_revenue.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(fig_selected_revenue, use_container_width=True)


    st.subheader('Ranking of Top-Performing Dealers Based on Total Sales')
    fig_ranking = px.bar(
        top_dealers,
        x='Total_Sales',
        y='Dealer_Name',
        title='Top-Performing Dealers Based on Total Sales',
        labels={'Total_Sales': 'Total Sales', 'Dealer_Name': 'Dealer Name'},
        color='Total_Sales',
        orientation='h',
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig_ranking.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(fig_ranking, use_container_width=True)
    
    # Regional performance
    st.subheader('Regional Analysis of Dealer Performance')
    fig_regional = px.bar(
        regional_performance,
        x='Dealer_Region',
        y='Total_Sales',
        title='Total Sales by Dealer Region',
        labels={'Total_Sales': 'Total Sales', 'Dealer_Region': 'Dealer Region'},
        color='Total_Sales',
        color_continuous_scale=px.colors.sequential.RdBu
    )
    fig_regional.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(fig_regional, use_container_width=True)
    
# TASK3 Cohort Analysis
def cohort_analysis():
    st.markdown("<div class='main-header'>Cohort Analysis & Car Sales Volume</div>", unsafe_allow_html=True)

    # Preprocessing for monthly cohort analysis
    cohort_data = data.copy()
    cohort_data.rename(columns={'Model': 'Car_Model', 'Price ($)': 'Price'}, inplace=True)

    # Monthly cohort aggregation
    monthly_model = cohort_data.groupby(['Year', 'Month', 'Car_Model']).agg(
        Total_Sales=('Car_id', 'count'),
        Avg_Price=('Price', 'mean')
    ).reset_index()
    monthly_model['Date'] = pd.to_datetime(monthly_model[['Year', 'Month']].assign(DAY=1))
    monthly_model.sort_values(by=['Car_Model', 'Date'], inplace=True)

    # Price elasticity
    monthly_model['Price_Change_%'] = monthly_model.groupby('Car_Model')['Avg_Price'].pct_change()
    monthly_model['Sales_Change_%'] = monthly_model.groupby('Car_Model')['Total_Sales'].pct_change()
    
    # Only calculate elasticity where price change is not zero or NaN
    monthly_model['Elasticity'] = np.where(
        (monthly_model['Price_Change_%'].notna()) & (monthly_model['Price_Change_%'] != 0),
        monthly_model['Sales_Change_%'] / monthly_model['Price_Change_%'],
        np.nan
    )

    # Average elasticity per model, only for valid elasticity values
    avg_elasticity = monthly_model.groupby('Car_Model')['Elasticity'].mean().reset_index()
    avg_elasticity = avg_elasticity.dropna()  
    avg_elasticity.rename(columns={'Elasticity': 'Avg_Elasticity'}, inplace=True)

    # Recommendation logic
    def recommend(elasticity):
        if elasticity < -1:
            return "Lower Price to Increase Sales"
        elif elasticity > -0.5:
            return "Consider Raising Price"
        else:
            return "Price Change Likely Ineffective"

    monthly_model = pd.merge(monthly_model, avg_elasticity, on='Car_Model', how='left')
    monthly_model['Recommendation'] = monthly_model['Avg_Elasticity'].apply(lambda x: recommend(x) if pd.notna(x) else "Insufficient Data")

    # Average Elasticity Table
    st.subheader("Average Price Elasticity by Car Model")
    st.dataframe(avg_elasticity.sort_values(by='Avg_Elasticity').reset_index(drop=True))

    selected_model = st.selectbox("Select a Car Model (Cohort Analysis)", monthly_model['Car_Model'].unique())
    filtered = monthly_model[monthly_model['Car_Model'] == selected_model]

    # Sales trend
    st.subheader(f"Total Sales Over Time - {selected_model}")
    fig_sales_trend = px.line(filtered, x='Date', y='Total_Sales', markers=True, title="Sales Trend")
    fig_sales_trend.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(fig_sales_trend, use_container_width=True)
    
    # Price trend
    st.subheader(f"Average Price Over Time - {selected_model}")
    fig_price_trend = px.line(filtered, x='Date', y='Avg_Price', markers=True, title="Price Trend")
    fig_price_trend.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600, 
    )
    st.plotly_chart(fig_price_trend, use_container_width=True)

    # Elasticity trend
    st.subheader(f"Elasticity Over Time - {selected_model}")
    fig_elasticity_trend = px.line(filtered, x='Date', y='Elasticity', markers=True, title="Elasticity of Demand")
    fig_elasticity_trend.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    st.plotly_chart(fig_elasticity_trend, use_container_width=True)

    # Recommendation
    latest = filtered.iloc[-1] if not filtered.empty else None
    if latest is not None:
        st.subheader("Price Adjustment Recommendation")
        recommendation_col1, recommendation_col2, recommendation_col3 = st.columns(3)
        with recommendation_col1:
            st.info(f"**Model:** {selected_model}")
        with recommendation_col2:
            avg_elasticity_value = latest['Avg_Elasticity']
            if pd.notna(avg_elasticity_value):
                st.info(f"**Average Elasticity:** {avg_elasticity_value:.2f}")
            else:
                st.info("**Average Elasticity:** Insufficient Data")
        with recommendation_col3:
            st.info(f"**Recommendation:** {latest['Recommendation']}")

# TASK4 Product Profitability
def product_profitability():
    st.markdown("<div class='main-header'>Car Models Revenue and Sales Volume Dashboard</div>", unsafe_allow_html=True)

    # Sales and Revenue by Model
    model_performance = data.groupby('Model').agg(
        Total_Sales=('Car_id', 'count'),
        Total_Revenue=('Price ($)', 'sum')
    ).reset_index()


    top_models = model_performance.sort_values(by='Total_Revenue', ascending=False)

    # Summary Table
    st.subheader("Car Models Summary")
    st.dataframe(top_models.reset_index(drop=True))

    # Revenue Chart
    st.subheader("Total Revenue by Car Model")
    fig_revenue = px.bar(
        top_models,
        x='Model',
        y='Total_Revenue',
        color='Total_Revenue',
        color_continuous_scale=px.colors.sequential.Tealgrn,
        title="Total Revenue by Car Model"
    )
    fig_revenue.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600, 
    )
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Sales Volume Chart
    st.subheader("Total Sales Volume by Car Model")
    fig_sales_bar = px.bar(
        top_models,
        x='Model',
        y='Total_Sales',
        color='Total_Sales',
        color_continuous_scale=px.colors.sequential.Blues,
        title="Total Sales Volume by Car Model"
    )
    fig_sales_bar.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600, 
    )
    st.plotly_chart(fig_sales_bar, use_container_width=True)

    model_selection = st.selectbox("Select a Car Model (Summary View)", model_performance['Model'].unique())
    selected_summary = model_performance[model_performance['Model'] == model_selection]
    
    st.subheader(f"Performance of {model_selection}")
    st.metric("Total Sales", f"{selected_summary['Total_Sales'].values[0]}")
    st.metric("Total Revenue", f"${selected_summary['Total_Revenue'].values[0]:,.2f}")

# TASK5 Strategic Recommendations
def strategic_recommendations():
    st.markdown("<div class='main-header'>Strategic Recommendations for Product and Dealer Partnerships</div>", unsafe_allow_html=True)

    # Total Sales by Company 
    company_sales = data.groupby('Company').agg(
        Total_Sales=('Price ($)', 'sum')
    ).reset_index().sort_values(by='Company') 

    fig_company_sales = px.bar(
        company_sales,
        x='Company',
        y='Total_Sales',
        title='Total Sales by Company',
        labels={'Company': 'Company', 'Total_Sales': 'Total Sales ($)'},
        color='Total_Sales',
        color_continuous_scale='Viridis',
        category_orders={'Company': company_sales['Company'].tolist()}  
    )

    fig_company_sales.update_layout(
        xaxis_tickangle=-45,
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )
    fig_company_sales.update_xaxes(tickmode='linear')

    st.subheader('Total Sales by Company')
    st.plotly_chart(fig_company_sales, use_container_width=True)

    # Average Price vs Sales Volume
    st.subheader('Average Price vs Sales Volume by Company')
    company_stats = data.groupby('Company').agg(
        Average_Price=('Price ($)', 'mean'),
        Sales_Volume=('Car_id', 'count')
    ).reset_index().sort_values(by='Company')

    fig_scatter = px.scatter(
        company_stats,
        x='Average_Price',
        y='Sales_Volume',
        color='Company',
        hover_name='Company',
        title='Average Price vs Sales Volume by Company',
        labels={
            'Average_Price': 'Average Price ($)',
            'Sales_Volume': 'Sales Volume'
        },
        size_max=60
    )

    fig_scatter.update_layout(
        height=600, 
        modebar_remove=['autoScale2d', 'zoom2d'],
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Price vs Sales Volume by Model
    st.subheader('Average Price vs Sales Volume by Car Model')
    model_stats = data.groupby('Model').agg(
        Average_Price=('Price ($)', 'mean'),
        Sales_Volume=('Car_id', 'count')
    ).reset_index()

    fig_price_model = px.scatter(
        model_stats,
        x='Average_Price',
        y='Sales_Volume',
        color='Model',
        hover_name='Model',
        title=' Average Price vs. Sales Volume by Car Model',
        labels={
            'Average_Price': 'Average Price ($)',
            'Sales_Volume': 'Sales Volume'
        }
    )

    fig_price_model.update_layout(
        height=600, 
        modebar_remove=['autoScale2d', 'zoom2d'],
    )

    st.plotly_chart(fig_price_model, use_container_width=True)
    
    # Sales Performance by Color
    st.subheader('Sales Performance by Color')
    color_sales = data.groupby('Color').agg(
        Total_Sales=('Price ($)', 'sum')
    ).reset_index().sort_values(by='Total_Sales', ascending=False)

    fig_color_sales = px.bar(
        color_sales,
        x='Color',
        y='Total_Sales',
        title='Sales Performance by Color',
        labels={'Color': 'Car Color', 'Total_Sales': 'Total Sales ($)'},
        color='Total_Sales',
        color_continuous_scale='YlOrRd_r'  
    )
    fig_color_sales.update_layout(
        modebar_remove=['autoScale2d', 'zoom2d'],
        height=600,  
    )

    st.plotly_chart(fig_color_sales, use_container_width=True)

    # Key Recommendations Section
    st.markdown("<div class='section-header'>Key Recommendations</div>", unsafe_allow_html=True)
    
    recommendations_col1, recommendations_col2 = st.columns(2)
    
    with recommendations_col1:
        st.subheader("Product Recommendations:")
        st.markdown("""
        1. **Focus on high-revenue models:** Allocate more marketing budget to models generating highest revenue.
        2. **Price optimization strategy:** Adjust pricing based on elasticity data to maximize profit margins.
        3. **Color preference targeting:** Promote car colors with highest sales performance.
        4. **Body style diversification:** Expand inventory of popular body styles while limiting less popular options.
        5. **Seasonal promotions:** Implement targeted promotions during months with historically lower sales volume.
        """)
    
    with recommendations_col2:
        st.subheader("Dealer Strategies:")
        st.markdown("""
        1. **Performance-based incentives:** Implement tiered reward system for high-performing dealers.
        2. **Regional focus:** Allocate additional resources to underperforming regions to boost sales.
        3. **Sales training program:** Develop specialized training for dealers with below-average sales metrics.
        4. **Cross-dealer collaboration:** Create knowledge-sharing platform between top and bottom performers.
        5. **Customer experience improvement:** Standardize excellent customer service practices from top dealers.
        """)
    
    st.subheader("Company Strategies:")
    st.markdown("""
    1. **Strategic partnerships:** Form alliances with companies showing strong market performance for co-branding opportunities.
    2. **Market segmentation:** Develop targeted marketing strategies based on customer segment preferences.
    3. **Product portfolio optimization:** Invest in expanding model lines with highest profit margins.
    4. **Data-driven inventory management:** Use sales trends to optimize inventory levels across dealers.
    5. **Customer loyalty programs:** Implement targeted rewards for repeat customers based on segmentation data.
    6. **Digital transformation initiative:** Invest in online sales platforms for dealers with younger customer demographics.
    7. **Sustainability focus:** Highlight fuel efficiency for segments where this is a purchasing factor.
    8. **Financing flexibility:** Offer customized financing options based on income segment analysis.
    """)


if st.session_state.current_page == "📊 Dashboard Overview":
    dashboard_overview()
elif st.session_state.current_page == "👥 Customer Segmentation":
    customer_segmentation()
elif st.session_state.current_page == "🏢 Dealer Performance":
    dealer_performance()
elif st.session_state.current_page == "📈 Cohort Analysis":
    cohort_analysis()
elif st.session_state.current_page == "💰 Product Profitability":
    product_profitability()
elif st.session_state.current_page == "🎯 Strategic Recommendations":
    strategic_recommendations()
