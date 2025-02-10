import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_models():
    """Load the PCA model and create scaler"""
    try:
        pca = joblib.load('pca_model.pkl')
        scaler = StandardScaler()
        return pca, scaler
    except:
        st.error("⚠️ Error: Could not load PCA model. Make sure 'pca_model.pkl' exists in the current directory.")
        return None, None

def create_feature_input(feature_names):
    """Create organized input sections for features in three parallel columns"""
    st.subheader("📊 Input Features")
    
    # Create three columns for parallel layout
    col1, col2, col3 = st.columns(3)
    input_values = {}
    
    # Group features into categories
    chemical_properties = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium"]
    visual_properties = ["Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
    phenolic_compounds = ["Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins"]
    
    # Chemical Properties Column
    with col1:
        st.markdown("##### 🧪 Chemical Properties")
        st.markdown("---")
        for feature in chemical_properties:
            input_values[feature] = st.slider(
                feature,
                min_value=0.0,
                max_value=15.0,
                value=7.0,
                step=0.1,
                help=f"Adjust the value for {feature}"
            )
    
    # Visual Properties Column
    with col2:
        st.markdown("##### 👁️ Visual Properties")
        st.markdown("---")
        for feature in visual_properties:
            input_values[feature] = st.slider(
                feature,
                min_value=0.0,
                max_value=15.0,
                value=7.0,
                step=0.1,
                help=f"Adjust the value for {feature}"
            )
    
    # Phenolic Compounds Column
    with col3:
        st.markdown("##### 🧬 Phenolic Compounds")
        st.markdown("---")
        for feature in phenolic_compounds:
            input_values[feature] = st.slider(
                feature,
                min_value=0.0,
                max_value=15.0,
                value=7.0,
                step=0.1,
                help=f"Adjust the value for {feature}"
            )
    
    return input_values

def main():
    # Page configuration
    st.set_page_config(page_title="Wine PCA Analysis", page_icon="🍷", layout="wide")
    
    # Title and introduction
    st.title("🍷 Interactive Wine PCA Analysis")
    st.markdown("""
    This application performs Principal Component Analysis (PCA) on wine chemical properties.
    Adjust the sliders to see how different feature values affect the PCA transformation.
    """)
    
    # Load models
    pca, scaler = load_models()
    if not pca or not scaler:
        return
    
    # Feature names
    feature_names = [
        "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
        "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
        "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
    ]
    
    # Create input interface
    input_values = create_feature_input(feature_names)
    
    # Center the Calculate PCA button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        calculate_button = st.button("🔄 Calculate PCA", type="primary", use_container_width=True)
    
    if calculate_button:
        # Convert input to array
        input_array = np.array(list(input_values.values())).reshape(1, -1)
        
        # Standardize and transform
        input_scaled = scaler.fit_transform(input_array)
        pca_transformed = pca.transform(input_scaled)
        
        # Create columns for results
        st.markdown("---")
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.subheader("📈 PCA Components")
            # Create a DataFrame for better presentation
            pca_df = pd.DataFrame(
                pca_transformed,
                columns=[f"PC{i+1}" for i in range(pca_transformed.shape[1])]
            )
            st.dataframe(pca_df.style.format("{:.4f}"))
            
            # Show explained variance
            st.markdown("##### Explained Variance Ratio")
            explained_variance = pd.DataFrame({
                'Component': [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                'Explained Variance (%)': pca.explained_variance_ratio_ * 100
            })
            st.dataframe(explained_variance.style.format({'Explained Variance (%)': "{:.2f}"}))
        
        with results_col2:
            st.subheader("🎯 2D Visualization")
            # Create an interactive scatter plot
            chart_data = pd.DataFrame({
                'PC1': [pca_transformed[0, 0]],
                'PC2': [pca_transformed[0, 1]]
            })
            st.scatter_chart(
                data=chart_data,
                x='PC1',
                y='PC2',
                size=200
            )

if __name__ == "__main__":
    main()