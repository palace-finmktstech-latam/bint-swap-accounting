import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.title("My First Streamlit App")

# Add a sidebar
st.sidebar.header("Settings")
user_name = st.sidebar.text_input("Your Name", "Guest")
color = st.sidebar.selectbox("Favorite Color", ["Blue", "Green", "Red", "Purple"])

# Display a greeting
st.write(f"Hello {user_name}! Welcome to your Streamlit dashboard.")
st.write(f"Your favorite color is {color}.")

# Create a simple chart
st.subheader("Sample Data Visualization")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)
st.line_chart(chart_data)

# Add an interactive element
if st.button("Generate Random Number"):
    st.write(f"Your random number is: {np.random.randint(1, 100)}")

# Add a file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())