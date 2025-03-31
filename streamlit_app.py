import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Configuration
st.set_page_config(page_title="Sri Krisha cafe Sales Dashboard", page_icon="🍽️", layout="wide")

# Enhanced Parsing and Data Loading Functions
def parse_date(date_str):
    try:
        if pd.isna(date_str) or date_str == 'UNKNOWN':
            return pd.NaT
        
        date_formats = [
            '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d', 
            '%d/%m/%Y', '%m/%d/%Y'
        ]
        
        for fmt in date_formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        return pd.NaT
    except:
        return pd.NaT

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = [col.strip() for col in df.columns]
        
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = df['Transaction Date'].apply(parse_date)
        
        df = df.dropna(subset=['Transaction Date'])
        
        numeric_columns = ['RATE', 'Quantity', 'AMOUNT']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_columns)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Train Sales Prediction Model
def train_sales_prediction_model(df):
    # Feature Engineering
    df['Month'] = df['Transaction Date'].dt.month
    df['DayOfWeek'] = df['Transaction Date'].dt.dayofweek
    
    # Encoding categorical variables
    label_encoders = {}
    categorical_columns = ['PRODUCT', 'Branch', 'Payment Method']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Select features
    features = [
        'Month', 'DayOfWeek', 
        'PRODUCT_encoded', 'Branch_encoded', 
        'Payment Method_encoded', 
        'Quantity', 'RATE'
    ]
    
    X = df[features]
    y = df['AMOUNT']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.sidebar.subheader("🤖 Prediction Model Performance")
    st.sidebar.write(f"Mean Absolute Error: ₹{mae:.2f}")
    st.sidebar.write(f"Mean Squared Error: {mse:.2f}")
    
    return rf_model, scaler, label_encoders

# Main App Function
def main():
    st.title("🍽️ Sri Krisha Cafe Sales Dashboard")
    st.markdown("### Upload your sales data for comprehensive analysis")

    # File Upload
    st.sidebar.header("📂 Upload Excel File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an Excel file", type=['xlsx', 'xls']
    )

    if uploaded_file is not None:
        # Load Data
        df = load_data(uploaded_file)
        
        if df is not None and not df.empty:
            # Success message
            st.success("Data loaded successfully!")
            
            # Sidebar for filters
            st.sidebar.header("🔍 Filters")

            # Dynamic filters based on available columns
            filter_columns = {
                'PRODUCT': 'Product',
                'Branch': 'Branch',
                'Payment Method': 'Payment Method'
            }

            filters = {}
            for col, label in filter_columns.items():
                if col in df.columns:
                    filters[col] = st.sidebar.multiselect(
                        f"Select {label}",
                        options=df[col].unique(),
                        default=df[col].unique()
                    )

            # Date range filter
            min_date = df['Transaction Date'].min().date()
            max_date = df['Transaction Date'].max().date()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            # Apply filters
            filtered_df = df.copy()
            for col, selected_values in filters.items():
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
            
            filtered_df = filtered_df[
                (filtered_df['Transaction Date'].dt.date >= start_date) &
                (filtered_df['Transaction Date'].dt.date <= max_date)
            ]
            
            # Show number of filtered records
            st.markdown(f"*Showing analysis for {len(filtered_df):,} records*")

            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_sales = filtered_df['AMOUNT'].sum()
                st.metric(label="💰 Total Sales", value=f"₹{total_sales:,.0f}")

            with col2:
                total_quantity = filtered_df['Quantity'].sum()
                st.metric(label="🛒 Total Quantity Sold", value=f"{total_quantity:,}")

            with col3:
                avg_price = filtered_df['RATE'].mean()
                st.metric(label="💲 Average Price", value=f"₹{avg_price:.2f}")

            with col4:
                unique_products = filtered_df['PRODUCT'].nunique()
                st.metric(label="🍽️ Unique Products", value=unique_products)

            # Train Prediction Model
            prediction_model, scaler, label_encoders = train_sales_prediction_model(df)

            # Prediction Section
            st.sidebar.header("🔮 Sales Prediction")
            
            # Get prediction inputs from sidebar
            prediction_product = st.sidebar.selectbox(
                "Select Product", df['PRODUCT'].unique()
            )
            prediction_branch = st.sidebar.selectbox(
                "Select Branch", df['Branch'].unique()
            )
            prediction_payment = st.sidebar.selectbox(
                "Select Payment Method", df['Payment Method'].unique()
            )
            prediction_month = st.sidebar.slider(
                "Select Month", 1, 12, 6
            )
            prediction_quantity = st.sidebar.number_input(
                "Quantity", min_value=1, max_value=20, value=1
            )
            prediction_rate = st.sidebar.number_input(
                "Rate (₹)", min_value=50, max_value=500, value=200
            )

            # Prepare prediction input
            prediction_input = np.array([
                prediction_month, 
                prediction_month % 7,  # Day of week approximation
                label_encoders['PRODUCT'].transform([prediction_product])[0],
                label_encoders['Branch'].transform([prediction_branch])[0],
                label_encoders['Payment Method'].transform([prediction_payment])[0],
                prediction_quantity,
                prediction_rate
            ]).reshape(1, -1)

            # Scale input
            prediction_input_scaled = scaler.transform(prediction_input)
            
            # Predict
            predicted_sales = prediction_model.predict(prediction_input_scaled)[0]
            
            st.sidebar.metric(
                label="🔮 Predicted Sales", 
                value=f"₹{predicted_sales:.2f}"
            )

            # Custom CSS for dashboard layout
            st.markdown("""
            <style>
            .visualization-container {
                background-color: #f9f9f9;
                border-radius: 10px;
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .section-title {
                padding: 10px;
                background-color: #4e8df5;
                color: white;
                border-radius: 5px;
                margin-bottom: 15px;
                text-align: center;
                font-weight: bold;
            }
            .stMetric {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                border: 1px solid #ddd;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            .stMetric > div {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # VISUALIZATION SECTIONS ORGANIZED IN 3 COLUMNS
            
            # Section 1: Sales Analysis
            st.markdown('<div class="section-title">📊 Sales Analysis</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### 🍔 Sales by Product")
                product_sales = filtered_df.groupby('PRODUCT')['AMOUNT'].sum().sort_values(ascending=False)
                fig_product_sales = px.bar(
                    product_sales, 
                    x=product_sales.index, 
                    y=product_sales.values, 
                    title="Total Sales by Product",
                    labels={'x': 'Product', 'y': 'Total Sales'}
                )
                st.plotly_chart(fig_product_sales, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### 🏪 Sales by Branch")
                branch_sales = filtered_df.groupby('Branch')['AMOUNT'].sum().sort_values(ascending=False)
                fig_branch_sales = px.pie(
                    branch_sales, 
                    values=branch_sales.values, 
                    names=branch_sales.index, 
                    title="Sales by Branch"
                )
                st.plotly_chart(fig_branch_sales, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### 💳 Payment Method")
                payment_sales = filtered_df.groupby('Payment Method')['AMOUNT'].sum()
                fig_payment = px.bar(
                    payment_sales, 
                    x=payment_sales.index, 
                    y=payment_sales.values, 
                    title="Sales by Payment Method",
                    labels={'x': 'Payment Method', 'y': 'Total Sales'}
                )
                st.plotly_chart(fig_payment, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 2: Time Analysis
            st.markdown('<div class="section-title">📈 Time Analysis</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### Monthly Trend")
                monthly_sales = filtered_df.groupby(pd.Grouper(key='Transaction Date', freq='M'))['AMOUNT'].sum()
                fig_time_series = px.line(
                    monthly_sales, 
                    x=monthly_sales.index, 
                    y=monthly_sales.values, 
                    title="Monthly Sales",
                    labels={'x': 'Month', 'y': 'Total Sales'}
                )
                st.plotly_chart(fig_time_series, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                # Extract day of week and hour from transaction date
                filtered_df['Day of Week'] = filtered_df['Transaction Date'].dt.day_name()
                filtered_df['Hour'] = filtered_df['Transaction Date'].dt.hour
                
                st.markdown("### Sales Heatmap")
                # Create heatmap data
                heatmap_data = filtered_df.groupby(['Day of Week', 'Hour'])['AMOUNT'].sum().unstack(fill_value=0)
                
                # Reorder days of week
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = heatmap_data.reindex(day_order)
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Sales"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale="Viridis",
                    title="Sales by Day & Hour"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                # Create a new column for weekday/weekend
                filtered_df['Day Type'] = filtered_df['Day of Week'].apply(
                    lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
                )
                
                st.markdown("### Weekday vs Weekend")
                # Group by day type
                day_type_sales = filtered_df.groupby('Day Type')['AMOUNT'].sum()
                
                # Create a pie chart
                fig_day_type = px.pie(
                    day_type_sales,
                    values=day_type_sales.values,
                    names=day_type_sales.index,
                    title="Weekday vs Weekend Sales",
                    color_discrete_map={'Weekday': '#636EFA', 'Weekend': '#EF553B'}
                )
                st.plotly_chart(fig_day_type, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 3: Product Analysis
            st.markdown('<div class="section-title">🍲 Product Analysis</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### Product Performance")
                
                # Top 5 products by sales
                top_products = filtered_df.groupby('PRODUCT')['AMOUNT'].sum().nlargest(5).sort_values(ascending=True)
                fig_top_products = px.bar(
                    top_products,
                    x=top_products.values,
                    y=top_products.index,
                    orientation='h',
                    title="Top 5 Products by Sales",
                    labels={'x': 'Total Sales', 'y': 'Product'}
                )
                st.plotly_chart(fig_top_products, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### Product Correlation")
                
                # Product and Payment Method Correlation
                product_payment_pivot = pd.pivot_table(
                    filtered_df, 
                    values='AMOUNT', 
                    index='PRODUCT', 
                    columns='Payment Method', 
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig_heatmap_corr = px.imshow(
                    product_payment_pivot,
                    labels=dict(x="Payment Method", y="Product", color="Sales"),
                    x=product_payment_pivot.columns,
                    y=product_payment_pivot.index,
                    color_continuous_scale="Blues",
                    title="Product vs Payment Method"
                )
                st.plotly_chart(fig_heatmap_corr, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                
                st.markdown("### Seasonal Products")
                filtered_df['Month'] = filtered_df['Transaction Date'].dt.month_name()
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
                filtered_df['Month'] = pd.Categorical(filtered_df['Month'], categories=month_order, ordered=True)
                top3_products = filtered_df.groupby('PRODUCT')['AMOUNT'].sum().nlargest(3).index.tolist()
                seasonal_df = filtered_df[filtered_df['PRODUCT'].isin(top3_products)]
                product_month_sales = seasonal_df.groupby(['PRODUCT', 'Month'], observed=True)['AMOUNT'].sum().reset_index()
                # Group by product and month
                product_month_sales = product_month_sales.sort_values(by=['Month'])
                fig_seasonal_products = px.line(
                product_month_sales,
                x='Month',
                y='AMOUNT',
                color='PRODUCT',
                markers=True,  # Adds markers for better readability
                title="Top Products by Month",
                labels={'Month': 'Month', 'AMOUNT': 'Sales', 'PRODUCT': 'Product'},
                category_orders={"Month": month_order}  # Ensures correct month order
                )
                # Create line chart for seasonal patterns
                fig_seasonal_products.update_traces(line=dict(width=2))  # Adjust line thickness
                fig_seasonal_products.update_layout(
                xaxis_title="Month",
                yaxis_title="Sales Amount",
                xaxis=dict(tickmode='array', tickvals=month_order),
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_seasonal_products, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Section 4: Performance Metrics
            st.markdown('<div class="section-title">💹 Performance Metrics</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                # Top selling products
                top_products = filtered_df.groupby('PRODUCT')['AMOUNT'].sum().nlargest(5)
                st.metric("Top Selling Product", top_products.index[0], f"₹{top_products.values[0]:,.2f}")
                
                # Average transaction value
                avg_transaction = filtered_df['AMOUNT'].mean()
                st.metric("Average Transaction", f"₹{avg_transaction:,.2f}")
                
                # Best performing branch
                top_branch = filtered_df.groupby('Branch')['AMOUNT'].sum().nlargest(1)
                st.metric("Top Branch", top_branch.index[0], f"₹{top_branch.values[0]:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### Seasonal Pattern")
                # Seasonal sales by month
                monthly_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                'July', 'August', 'September', 'October', 'November', 'December']
                
                # Group by month and calculate total sales
                month_sales = filtered_df.groupby('Month')['AMOUNT'].sum().reindex(monthly_order)
                
                fig_seasonal = px.line(
                    month_sales,
                    x=month_sales.index,
                    y=month_sales.values,
                    markers=True,
                    title="Monthly Sales Pattern",
                    labels={'x': 'Month', 'y': 'Total Sales'}
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                st.markdown("### Product Comparison")
                
                # Multi-select for products (limit to 3 for clarity)
                selected_products = st.multiselect(
                    "Select products to compare (max 3):",
                    options=filtered_df['PRODUCT'].unique(),
                    default=filtered_df['PRODUCT'].unique()[:2] if len(filtered_df['PRODUCT'].unique()) >= 2 else filtered_df['PRODUCT'].unique()
                )
                
                if selected_products:
                    # Limit to 3 products for better visualization
                    selected_products = selected_products[:3]
                    
                    # Filter data for selected products
                    product_comparison_df = filtered_df[filtered_df['PRODUCT'].isin(selected_products)]
                    
                    # Group by product and date
                    product_time_series = product_comparison_df.groupby(['PRODUCT', pd.Grouper(key='Transaction Date', freq='M')])['AMOUNT'].sum().reset_index()
                    
                    # Create line chart
                    fig_product_comparison = px.line(
                        product_time_series,
                        x='Transaction Date',
                        y='AMOUNT',
                        color='PRODUCT',
                        title="Product Sales Comparison",
                        labels={'Transaction Date': 'Month', 'AMOUNT': 'Sales'}
                    )
                    st.plotly_chart(fig_product_comparison, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Customer Analysis Section (if data available)
            if 'Customer Type' in filtered_df.columns or 'Rating' in filtered_df.columns:
                st.markdown('<div class="section-title">👥 Customer Analysis</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    st.markdown("### Customer Type")
                    if 'Customer Type' in filtered_df.columns:
                        customer_type_counts = filtered_df['Customer Type'].value_counts()
                        fig_customer_type = px.pie(
                            customer_type_counts,
                            values=customer_type_counts.values,
                            names=customer_type_counts.index,
                            title="Customer Types",
                            hole=0.4  # Makes it a donut chart
                        )
                        st.plotly_chart(fig_customer_type, use_container_width=True)
                    else:
                        st.info("No 'Customer Type' column found")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    st.markdown("### Customer Ratings")
                    if 'Rating' in filtered_df.columns:
                        rating_counts = filtered_df['Rating'].value_counts().sort_index()
                        fig_ratings = px.bar(
                            rating_counts,
                            x=rating_counts.index,
                            y=rating_counts.values,
                            title="Ratings Distribution",
                            labels={'x': 'Rating', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_ratings, use_container_width=True)
                    else:
                        st.info("No 'Rating' column found")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col3:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    # Sales Funnel Analysis (if data available)
                    if 'Transaction Status' in filtered_df.columns:
                        st.markdown("### Sales Funnel")
                        status_counts = filtered_df['Transaction Status'].value_counts()
                        fig_funnel = px.funnel(
                            status_counts,
                            x=status_counts.values,
                            y=status_counts.index,
                            title="Transaction Status Funnel"
                        )
                        st.plotly_chart(fig_funnel, use_container_width=True)
                    elif 'Customer ID' in filtered_df.columns:
                        st.markdown("### Customer Spending")
                        customer_spending = filtered_df.groupby('Customer ID')['AMOUNT'].sum().sort_values(ascending=False)
                        
                        # Create spending buckets
                        spending_ranges = [0, 100, 500, 1000, 5000, float('inf')]
                        labels = ['₹0-₹100', '₹100-₹500', '₹500-₹1000', '₹1000-₹5000', '₹5000+']
                        
                        # Cut the spending into bins
                        spending_distribution = pd.cut(customer_spending, bins=spending_ranges, labels=labels, right=False)
                        spending_counts = spending_distribution.value_counts().sort_index()
                        
                        # Create a bar chart
                        fig_spending = px.bar(
                            spending_counts,
                            x=spending_counts.index,
                            y=spending_counts.values,
                            title="Customer Spending",
                            labels={'x': 'Spending Range', 'y': 'Customers'}
                        )
                        st.plotly_chart(fig_spending, use_container_width=True)
                    else:
                        st.info("No customer transaction data available")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # ROI Analysis Section (if data available)
            if 'Cost' in filtered_df.columns:
                st.markdown('<div class="section-title">💹 ROI Analysis</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    # Calculate profit and ROI
                    filtered_df['Profit'] = filtered_df['AMOUNT'] - filtered_df['Cost']
                    filtered_df['ROI'] = (filtered_df['Profit'] / filtered_df['Cost']) * 100
                    
                    # Product ROI
                    product_roi = filtered_df.groupby('PRODUCT').agg({
                        'AMOUNT': 'sum',
                        'Cost': 'sum',
                        'Profit': 'sum'
                    }).sort_values(by='Profit', ascending=False)
                    
                    product_roi['ROI%'] = (product_roi['Profit'] / product_roi['Cost'] * 100).round(2)
                    
                    # Create a bar chart for ROI by product
                    fig_roi = px.bar(
                        product_roi,
                        x=product_roi.index,
                        y='ROI%',
                        title="ROI by Product (%)",
                        labels={'x': 'Product', 'y': 'ROI (%)'}
                    )
                    st.plotly_chart(fig_roi, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    # Total Profit Metrics
                    total_revenue = filtered_df['AMOUNT'].sum()
                    total_cost = filtered_df['Cost'].sum()
                    total_profit = filtered_df['Profit'].sum()
                    overall_roi = (total_profit / total_cost * 100)
                    
                    st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
                    st.metric("Total Cost", f"₹{total_cost:,.2f}")
                    st.metric("Total Profit", f"₹{total_profit:,.2f}")
                    st.metric("Overall ROI", f"{overall_roi:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col3:
                    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
                    # Monthly Profit Trend
                    monthly_profit = filtered_df.groupby(pd.Grouper(key='Transaction Date', freq='M')).agg({
                        'AMOUNT': 'sum',
                        'Cost': 'sum',
                        'Profit': 'sum'
                    }).reset_index()
                    
                    monthly_profit['ROI%'] = (monthly_profit['Profit'] / monthly_profit['Cost'] * 100).round(2)
                    
                    fig_monthly_roi = px.line(
                        monthly_profit,
                        x='Transaction Date',
                        y=['AMOUNT', 'Cost', 'Profit'],
                        title="Monthly Revenue, Cost & Profit",
                        labels={'value': 'Amount (₹)', 'Transaction Date': 'Month', 'variable': 'Metric'}
                    )
                    st.plotly_chart(fig_monthly_roi, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Detailed Data Table 
            st.markdown('<div class="section-title">📋 Data Table</div>', unsafe_allow_html=True)
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
            st.dataframe(filtered_df, use_container_width=True)
            
            # Export option
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "filtered_sales_data.csv",
                "text/csv",
                key='download-csv'
            )

        else:
            st.error("Failed to load the data.")
    else:
        st.info("Please upload an Excel file to begin analysis.")

# Custom CSS
# Custom CSS
st.markdown("""
<style>
.visualization-container {
  background-color: #f9f9f9;
  border-radius: 10px;
  border: 1px solid #ddd;
  padding: 15px;
  margin-bottom: 15px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  color: #f0f0f0; /* Dark white */
}

.section-title {
  padding: 10px;
  background-color: #4e8df5;
  color: #e6e6e6; /* Light text color for section titles */
  border-radius: 5px;
  margin-bottom: 15px;
  text-align: center;
  font-weight: bold;
}

.stMetric {
  background-color: #f0f2f6;
  padding: 10px;
  border-radius: 5px;
  text-align: center;
  border: 1px solid #ddd;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.stMetric > div {
  display: flex;
  flex-direction: column;
  align-items: center;
  color: #f0f0f0; /* Dark white */
}

/* Global text color settings */
body {
  color: #f0f0f0 !important; /* Dark white */
}

/* Target only specific white background elements */
.stDataFrame, /* For dataframe white backgrounds */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
div[style*="background-color: white"],
div[style*="background-color: #fff"],
div[style*="background-color: rgb(255, 255, 255)"],
div[style*="background: white"],
div[style*="background: #fff"],
div[style*="background: rgb(255, 255, 255)"],
.stTile, /* Metric tiles */
.st-emotion-cache-q8sbsg p, /* New Streamlit metric value */ 
.st-emotion-cache-p5msec p /* New Streamlit metric label */ {
  color: #333333 !important; /* Dark text color */
}

/* Improved visibility for text on blue backgrounds */
div[style*="background-color: #4e8df5"],
div[style*="background-color: rgb(78, 141, 245)"],
div[style*="background: #4e8df5"],
div[style*="background: rgb(78, 141, 245)"],
.section-title,
div[style*="background-color: blue"],
div[style*="background: blue"],
/* Target specifically blue elements from your screenshots */
[data-testid="stMarkdownContainer"] p[style*="color: white"],
.stMarkdown div[style*="background-color: #0000ff"],
.stMarkdown div[style*="background-color: rgb(0, 0, 255)"],
.stMarkdown div[style*="background-color: blue"],
/* Other blue background cases */
div[style*="background-color: #4287f5"],
div[style*="background-color: #0066cc"] {
  color: white !important; /* Pure white text for better contrast on blue */
  font-weight: 600 !important; /* Make text slightly bolder */
  text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.5) !important; /* Add subtle text shadow for better readability */
}

/* BLACK BACKGROUND TEXT - ensure white text for black backgrounds */
div[style*="background-color: black"],
div[style*="background-color: #000"],
div[style*="background-color: rgb(0, 0, 0)"],
div[style*="background: black"],
div[style*="background: #000"],
div[style*="background: rgb(0, 0, 0)"],
div[style*="background-color: #1e1e1e"],
div[style*="background-color: #222"],
div[style*="background-color: #272727"],
/* Specific sections with dark background */
[data-testid="stHeader"],
[data-testid="baseButton-secondary"],
/* Specific metric sections */
[data-testid="stMarkdownContainer"]:has(h3:contains("Prediction Model Performance")),
div:contains("Showing analysis for 2,034 records"),
div:contains("Prediction Model Performance") {
  color: white !important; /* Bright white for best visibility on dark backgrounds */
  font-weight: 500 !important; /* Slightly bolder */
  text-shadow: 0px 0px 1px rgba(255, 255, 255, 0.2) !important; /* Subtle glow for legibility */
}

/* Specifically target metric cards */
.row-widget.stButton > button:has(div[style*="background-color: white"]),
.row-widget.stButton > button:has(div[style*="background-color: #fff"]) {
  color: #333333 !important;
}

/* Target text in cells with white background */
.cell-with-white-bg, /* If you have custom classes */
td[style*="background-color: white"],
td[style*="background-color: #fff"],
td[style*="background: white"],
td[style*="background: #fff"] {
  color: #333333 !important;
}

/* Chart text color */
.js-plotly-plot .plotly .gtitle, 
.js-plotly-plot .plotly .xtitle, 
.js-plotly-plot .plotly .ytitle {
  fill: #f0f0f0 !important; /* Dark white */
}

.js-plotly-plot .plotly .xtick text, 
.js-plotly-plot .plotly .ytick text {
  fill: #f0f0f0 !important; /* Dark white */
}

/* Special case for metric values showing in white cards like in your screenshots */
.element-container .stMarkdown div[data-testid="stMarkdownContainer"] p {
  color: #333333 !important; /* Dark text */
}

/* Target specifically the white cards in your dashboard */
[data-testid="column"] > div > div > div > div {
  color: #333333 !important; /* Dark text */
}

/* Target text in white tiles from your screenshots */
div[data-baseweb="card"],
div[data-testid="stVerticalBlock"] > div > div[style*="background-color: white"],
div[data-testid="stHorizontalBlock"] > div > div[style*="background-color: white"] {
  color: #333333 !important; /* Dark text */
}

/* DIRECT OVERRIDE for "Prediction Model Performance" and "Showing analysis" text */
h3:contains("Prediction Model Performance"),
div:contains("Prediction Model Performance"),
div:contains("Showing analysis for 2,034 records"),
.element-container:contains("Prediction Model Performance") *,
.element-container:contains("Showing analysis for") * {
  color: white !important;
  font-weight: 600 !important;
  text-shadow: 0px 0px 2px rgba(255, 255, 255, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)
# Run the main function
if __name__ == "__main__":
    main()
