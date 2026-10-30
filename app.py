import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Simple Linear Regression",
    page_icon="📈",
    # layout="wide"
)

# Title and description
st.title(" Simple Linear Regression Demo")
st.markdown("""
 **Simple Linear Regression**
""")

# Sidebar for user inputs
st.sidebar.header(" Controls")

# Option to use sample data or upload
data_option = st.sidebar.radio(
    "Choose data source:",
    ["Use Sample Dataset", "Upload Your Own CSV"]
)

# Initialize variables
X = None
y = None
feature_name = ""
target_name = ""

if data_option == "Use Sample Dataset":
    # Load sample datasets
    sample_choice = st.sidebar.selectbox(
        "Choose sample dataset:",
        ["Student Hours vs Scores", "House Size vs Price", "Advertising vs Sales"]
    )

    if sample_choice == "Student Hours vs Scores":
        # Generate sample data: Hours studied vs Exam scores
        np.random.seed(42)
        hours = np.random.uniform(1, 10, 50)
        scores = 25 + 5 * hours + np.random.normal(0, 5, 50)
        df = pd.DataFrame({'Hours_Studied': hours, 'Exam_Score': scores})
        feature_name = "Hours_Studied"
        target_name = "Exam_Score"
    
    elif sample_choice == "House Size vs Price":
        # Generate sample data: House size vs Price
        np.random.seed(42)
        size = np.random.uniform(500, 3000, 50)
        price = 50000 + 200 * size + np.random.normal(0, 50000, 50)
        df = pd.DataFrame({'House_Size_sqft': size, 'Price_USD': price})
        feature_name = "House_Size_sqft"
        target_name = "Price_USD"
        
    else:  # Advertising vs Sales
        # Generate sample data: Advertising spend vs Sales
        np.random.seed(42)
        ad_spend = np.random.uniform(100, 1000, 50)
        sales = 50 + 0.8 * ad_spend + np.random.normal(0, 30, 50)
        df = pd.DataFrame({'Advertising_Spend': ad_spend, 'Sales': sales})
        feature_name = "Advertising_Spend"
        target_name = "Sales"
        
    st.write(f" **Sample Dataset:** {sample_choice}")

else:  # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(" **Uploaded Dataset Preview:**")
    
        # Let user select columns
        columns = df.columns.tolist()
        col1, col2 = st.columns(2)
    
        with col1:
            feature_name= st.selectbox("Select Feature (x) column:", columns)
        with col2:
            target_name= st.selectbox("Select Target (y) column:", columns)
    else:
        st.info("Please upload a CSV file to begin")
        st.stop()
    
# Display dataset
st.subheader("Dataset Preview")
col1,col2 = st.columns([2,1])
with col1:
    st.dataframe(df.head(10), use_container_width=True)
with col2:
    st.metric("Total Samples", len(df))
    st.metric("Features", len(df.columns))
    
# Prepare data
x = df[[feature_name]].values
y = df[target_name].values

# Create two columns for main content
col1, col2 = st.columns([3,2])

with col1:
    st.subheader("Data Visualization")
    
    # Create scatter plot
    fig = go.Figure()
    
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df[feature_name],
        y=df[target_name],
        mode='markers',
        name='Data Points',
        marker=dict(size=10, opacity=0.7, color='blue')
    ))
    
    fig.update_layout(
        title=f"{feature_name} vs {target_name}",
        xaxis_title=feature_name,
        yaxis_title=target_name,
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    st.subheader("Model Configuration")
    
    # Train-Test split slider
    test_size = st.slider(
        "Test Set Size(%)",
        min_value=10,
        max_value=40,
        value=20,
        help= "Percentage of data to use for testing"
    )
    
    # Train model button
    if st.button("Trian Linear Regression Model", use_container_width=True):
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(
            x,y, test_size=test_size/100, random_state=42
        )
        
        # Create and train model
        model = LinearRegression()
        model.fit(x_train,y_train)
        
        # Make Prediction
        y_pred = model.predict(x_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results in session state
        st.session_state['model']=model
        st.session_state['x_train']= x_train
        st.session_state['y_train']= y_train
        st.session_state['x_test']=x_test
        st.session_state['y_test']=y_test
        st.session_state['y_pred']=y_pred
        st.session_state['mse']=mse
        st.session_state['r2']= r2
        
# Display results if model has been trained 
if 'model' in st.session_state:
    st.divider()
    st.subheader("Model Results")
            
    col1, col2, col3 = st.columns(3)
            
    with col1:
        st.metric("Model Intercept", f"{st.session_state['model'].intercept_:.2f}")
    with col2:
        st.metric("Model Slope", f"{st.session_state['model'].coef_[0]:.2f}")
    with col3:
        st.metric("R2 Score", f"{st.session_state['r2']:.3f}")
            
        # Create visualtization with regression line
        fig2 = go.Figure()
            
     # Trianing data
    fig2.add_trace(go.Scatter(
        x=st.session_state['x_train'].flatten(),
        y=st.session_state['y_train'],
        mode= 'markers',
        name='Trianing Data',
        marker=dict(size=8)
        ))
                          
    # Test data
    fig2.add_trace(go.Scatter(
        x=st.session_state['x_test'].flatten(),
        y=st.session_state['y_test'],
        mode= 'markers',
        name='Test Data(Actual)',
        marker=dict(color='green', size=10, opacity=0.8)
    ))
    
    # Prediction
    fig2.add_trace(go.Scatter(
        x=st.session_state['x_test'].flatten(),
        y=st.session_state['y_pred'],
        mode='lines+markers',
        name='Prediction',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Regression line for entire range
    x_range = np.array([[x.min()], [x.max()]])
    y_range_pred = st.session_state['model'].predict(x_range)
    
    fig2.add_trace(go.Scatter(
        x=x_range.flatten(),
        y=y_range_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='black', width=4, dash='dash')
    ))
    
    fig2.update_layout(
        title="Linear Regression Fit",
        xaxis_title= feature_name,
        yaxis_title=target_name,
        height=500,
        template='plotly_white',
        showlegend=True       
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Prediction section
    st.subheader("Make Prediction")
    col1,col2 = st.columns([2,3])
    
    with col1:
        input_value = st.number_input(
            f"Enter {feature_name} vlaue:",
            min_value=float(x.min()),
            max_value=float(x.max()),
            value=float(x.mean())
        )
        
        if st.button("Predict"):
            prediction = st.session_state['model'].predict([[input_value]])[0]
            st.success(f"Predicted {target_name}: **{prediction:.2f}**")
            
    with col2:
        # Equation display 
        st.info("### Regression Equation")
        intercept = st.session_state['model'].intercept_
        slope = st.session_state['model'].coef_[0]
        st.latex(f"y = {slope:.2f}x + {intercept:.2f}")
        st.caption(f"Where: y = {target_name}, x = {feature_name}")
        
# Explanation section
with st.expander("Learn about Linear Regression"):
    st.markdown("""
                ## What is Simplw Linear Regression?
                """)
# Footer
st.divider()
st.caption("Built with Streamlit | Simple Linear Regression Demo")

    
         
        