import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Streamlit UI
st.title('Cryptocurrency Price Analysis')

# File Upload Section
st.sidebar.header('Upload Your CSV Files')

uploaded_files = st.sidebar.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])

# Load Data
if uploaded_files:
    dfs = {}
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        dfs[uploaded_file.name] = df
    
    st.sidebar.write("### Available Datasets:")
    selected_file = st.sidebar.selectbox("Select a dataset", list(dfs.keys()))
    
    if selected_file:
        st.write(f"### Data Preview: {selected_file}")
        st.dataframe(dfs[selected_file].head())
        
        # Ensure the dataset has a 'date' column
        if 'date' in dfs[selected_file].columns:
            dfs[selected_file]['date'] = pd.to_datetime(dfs[selected_file]['date'])
            
            # Visualization: Price Trend Over Time
            plt.figure(figsize=(10, 5))
            plt.plot(dfs[selected_file]['date'], dfs[selected_file]['price_usd'], label='Price USD')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title(f'Price Trend for {selected_file}')
            plt.legend()
            st.pyplot(plt)
        else:
            st.error("The selected dataset does not have a 'date' column.")
else:
    st.warning("Please upload CSV files to proceed.")
