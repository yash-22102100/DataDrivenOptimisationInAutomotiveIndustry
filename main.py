


import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans


# Load the data
data = pd.read_csv('Car_Sales.csv')


# Data Cleaning
data.dropna(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])


# Sidebar for user input
st.sidebar.title("Customer Segmentation App")
st.sidebar.markdown("Explore customer demographics and purchasing behavior.")


# K-means Clustering
X = data[['Annual Income', 'Price ($)']]
kmeans = KMeans(n_clusters=4)  # Adjust based on your needs
data['Segment'] = kmeans.fit_predict(X)


# Summary of Customer Segments
summary = data.groupby('Segment').agg({
   'Customer Name': 'count',
   'Annual Income': 'mean',
   'Price ($)': 'mean'
}).reset_index()


st.subheader("Customer Segments Summary")
st.dataframe(summary)


# Gender Distribution
gender_fig = px.histogram(data, x='Gender', color='Gender',
                          color_discrete_map={'Male': 'blue', 'Female': 'pink'},
                          title='Gender Distribution of Customers')
st.plotly_chart(gender_fig)


# Annual Income Distribution
income_fig = px.histogram(data, x='Annual Income',
                          title='Annual Income Distribution',
                          nbins=30)
st.plotly_chart(income_fig)


# Purchase Trends Over Time
data['Month'] = data['Date'].dt.to_period('M')
purchase_trends = data.groupby('Month').size().reset_index(name='Number of Purchases')
purchase_trends['Month'] = purchase_trends['Month'].dt.to_timestamp()


trend_fig = px.line(purchase_trends, x='Month', y='Number of Purchases',
                   title='Purchase Trends Over Time')
st.plotly_chart(trend_fig)


# Add a title to the main page
st.title("Customer Segmentation and Insights")
st.markdown("Explore the customer segments and insights derived from the dataset.")
import pandas as pd
import plotly.express as px
import streamlit as st

# Load the data
data = pd.read_csv('Car_Sales.csv')

# Step 1: Data Preparation
data['Date'] = pd.to_datetime(data['Date'])

# Create additional columns for Year and Month
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Step 2: Performance Metrics Over Time
monthly_performance = data.groupby(['Year', 'Month', 'Dealer_Name']).agg(
    Total_Sales=('Car_id', 'count'),
    Total_Revenue=('Price ($)', 'sum')
).reset_index()

# Create a date column for easier plotting
monthly_performance['Date'] = pd.to_datetime(monthly_performance[['Year', 'Month']].assign(DAY=1))

# Calculate total sales and revenue for each dealer
dealer_performance = monthly_performance.groupby('Dealer_Name').agg(
    Total_Sales=('Total_Sales', 'sum'),
    Total_Revenue=('Total_Revenue', 'sum')
).reset_index()

# Rank dealers based on total sales
dealer_performance['Sales_Rank'] = dealer_performance['Total_Sales'].rank(ascending=False)
dealer_performance['Revenue_Rank'] = dealer_performance['Total_Revenue'].rank(ascending=False)

# Sort the dealers by total sales for display
top_dealers = dealer_performance.sort_values(by='Total_Sales', ascending=False)

# Regional Analysis
regional_performance = data.groupby('Dealer_Region').agg(
    Total_Sales=('Car_id', 'count'),
    Total_Revenue=('Price ($)', 'sum')
).reset_index()

# Step 3: Interactive Visualizations with Plotly

# Create an interactive line plot for Total Sales Over Time
fig_sales = px.line(
    monthly_performance,
    x='Date',
    y='Total_Sales',
    color='Dealer_Name',
    title='Total Sales Over Time by Dealer',
    labels={'Total_Sales': 'Total Sales', 'Date': 'Date'},
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set2  # Change color palette
)

# Create an interactive line plot for Total Revenue Over Time
fig_revenue = px.line(
    monthly_performance,
    x='Date',
    y='Total_Revenue',
    color='Dealer_Name',
    title='Total Revenue Over Time by Dealer',
    labels={'Total_Revenue': 'Total Revenue', 'Date': 'Date'},
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set2  # Change color palette
)

# Show the plots in a local Jupyter environment (for Colab, run in separate cells)
fig_sales.show()
fig_revenue.show()

# Create a bar chart for dealer rankings based on Total Sales
fig_ranking = px.bar(
    top_dealers,
    x='Total_Sales',
    y='Dealer_Name',
    title='Top-Performing Dealers Based on Total Sales',
    labels={'Total_Sales': 'Total Sales', 'Dealer_Name': 'Dealer Name'},
    color='Total_Sales',
    orientation='h',
    color_continuous_scale=px.colors.sequential.Blues  # Color gradient for the bars
)

# Create a bar chart for regional performance
fig_regional = px.bar(
    regional_performance,
    x='Dealer_Region',
    y='Total_Sales',
    title='Total Sales by Dealer Region',
    labels={'Total_Sales': 'Total Sales', 'Dealer_Region': 'Dealer Region'},
    color='Total_Sales',
    color_continuous_scale=px.colors.sequential.RdBu  # Change color palette
)

# Show the ranking bar chart
fig_ranking.show()

# Show the regional performance chart
fig_regional.show()

# Optional: Create a Streamlit App
st.title('Dealer Performance Dashboard')

# Display the ranking of top-performing dealers
st.subheader('Top-Performing Dealers')
st.dataframe(top_dealers[['Dealer_Name', 'Total_Sales', 'Total_Revenue', 'Sales_Rank', 'Revenue_Rank']])

# Dealer Selection
dealer_selection = st.selectbox("Select Dealer", monthly_performance['Dealer_Name'].unique())

# Filter data based on dealer selection
filtered_data = monthly_performance[monthly_performance['Dealer_Name'] == dealer_selection]

# Plot Total Sales for selected dealer
st.subheader(f'Total Sales for {dealer_selection} Over Time')
fig_selected_sales = px.line(
    filtered_data,
    x='Date',
    y='Total_Sales',
    title=f'Total Sales Over Time for {dealer_selection}',
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set2  # Change color palette
)

st.plotly_chart(fig_selected_sales)

# Plot Total Revenue for selected dealer
st.subheader(f'Total Revenue for {dealer_selection} Over Time')
fig_selected_revenue = px.line(
    filtered_data,
    x='Date',
    y='Total_Revenue',
    title=f'Total Revenue Over Time for {dealer_selection}',
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set2  # Change color palette
)

st.plotly_chart(fig_selected_revenue)

# Show the ranking bar chart in Streamlit
st.subheader('Ranking of Top-Performing Dealers Based on Total Sales')
st.plotly_chart(fig_ranking)

# Show the regional performance chart in Streamlit
st.subheader('Regional Analysis of Dealer Performance')
st.plotly_chart(fig_regional)

# Save this code in a .py file and run using: streamlit run your_script.py
