import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lagos Real Estate AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Cyberpunk Dark Mode) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b24; }
    /* Cards */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1F2937; border: 1px solid #374151;
        padding: 20px; border-radius: 10px;
    }
    /* Headers */
    h1, h2, h3 { color: #00ADB5 !important; font-family: 'Helvetica Neue', sans-serif; }
    /* Metrics */
    [data-testid="stMetricValue"] { color: #00ADB5; }
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('lagos_houses_prices.csv')
        # Filter for cleaner charts (10M - 2B)
        df = df[(df['cleaned_price'] >= 10_000_000) & 
                (df['cleaned_price'] <= 2_000_000_000) & 
                (df['Bedrooms'] >= 1)].copy()
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please upload 'lagos_houses_prices.csv'")
        return pd.DataFrame()

df = load_data()

# --- 2. LOAD SAVED MODEL (For Prediction) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('lagos_model.pkl')
        cols = joblib.load('model_features.pkl')
        return model, cols
    except FileNotFoundError:
        return None, []

prod_model, prod_features = load_model()

# --- NAVIGATION ---
st.sidebar.title("🏙️ Navigation")
page = st.sidebar.radio("Go to", ["📊 Market Dashboard", "🤖 AI Price Predictor"])

# ==========================================
# PAGE 1: MARKET DASHBOARD (Merged & Interactive)
# ==========================================
if page == "📊 Market Dashboard":
    st.title("📊 Lagos Real Estate Intelligence")
    st.markdown("---")
    
    if not df.empty:
        # --- SIDEBAR FILTERS (Only show on Dashboard) ---
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filter Data")
        
        # 1. Tier Filter
        tier_options = sorted(df['Neighborhood_Tier'].unique())
        selected_tiers = st.sidebar.multiselect(
            "Select Neighborhood Tier:",
            options=tier_options,
            default=tier_options
        )
        
        # 2. Type Filter
        type_options = sorted(df['Prop_Type'].unique())
        selected_types = st.sidebar.multiselect(
            "Select Property Types:",
            options=type_options,
            default=type_options
        )
        
        # Apply Filters to create a dynamic dataset
        df_filtered = df[
            (df['Neighborhood_Tier'].isin(selected_tiers)) & 
            (df['Prop_Type'].isin(selected_types))
        ]
        
        # --- ROW 1: KPI CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Listings", f"{len(df_filtered):,}")
        col2.metric("Median Price", f"₦{df_filtered['cleaned_price'].median():,.0f}")
        
        # Avoid crash if filter is empty
        if not df_filtered.empty:
            avg_price = df_filtered['cleaned_price'].mean()
            top_loc = df_filtered['Location'].mode()[0]
        else:
            avg_price = 0
            top_loc = "N/A"
            
        col3.metric("Avg Price", f"₦{avg_price:,.0f}")
        col4.metric("Top Location", top_loc)
        
        st.markdown("---")
        
        # --- ROW 2: DISTRIBUTION & CORRELATION ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("💰 Price Distribution")
            fig_hist = px.histogram(df_filtered, x="cleaned_price", nbins=50, 
                                    title="Where is the inventory concentrated?",
                                    color_discrete_sequence=['teal'], 
                                    template='plotly_dark', log_y=True)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            st.subheader("🔗 What drives the price?")
            corr_cols = ['cleaned_price', 'Bedrooms', 'dist_to_lekki', 'Neighborhood_Tier', 'Is_Island']
            corr_matrix = df_filtered[corr_cols].corr().round(2)
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 color_continuous_scale="RdBu_r", 
                                 title="Correlation Matrix", 
                                 template='plotly_dark')
            st.plotly_chart(fig_corr, use_container_width=True)

        # --- ROW 3: PRICE DRIVERS ---
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📍 Price by Neighborhood Tier")
            fig_tier = px.box(df_filtered, x='Neighborhood_Tier', y='cleaned_price', 
                              color='Neighborhood_Tier', 
                              template='plotly_dark', log_y=True,
                              title="The 'Location Multiplier'")
            st.plotly_chart(fig_tier, use_container_width=True)
            
        with c4:
            st.subheader("🏠 Price by Property Type")
            # Sort by median for better visual
            if not df_filtered.empty:
                type_order = df_filtered.groupby('Prop_Type')['cleaned_price'].median().sort_values().index
            else:
                type_order = []
                
            fig_type = px.box(df_filtered, x='Prop_Type', y='cleaned_price', 
                              color='Prop_Type', 
                              template='plotly_dark', log_y=True,
                              category_orders={'Prop_Type': type_order},
                              title="The 'Luxury Ladder'")
            st.plotly_chart(fig_type, use_container_width=True)
            
        # --- ROW 4: THE MATRIX ---
        st.subheader("💎 The Value Matrix")
        st.markdown("Median Price (₦) for every combination of Location and Size.")
        if not df_filtered.empty:
            pivot = df_filtered.pivot_table(index='Neighborhood_Tier', columns='Bedrooms', values='cleaned_price', aggfunc='median')
            fig_heat = px.imshow(pivot, text_auto=".2s", aspect="auto",
                                 color_continuous_scale="Viridis", origin='lower', 
                                 template='plotly_dark')
            st.plotly_chart(fig_heat, use_container_width=True)

        # --- ROW 5: GEOSPATIAL MAP ---
        st.subheader("🗺️ Listing Map")
        st.markdown("Zoom in to see clusters in Lekki vs Mainland.")
        # Filter map data for valid lat/lon
        map_df = df_filtered[(df_filtered['lat'] > 6.3) & (df_filtered['lat'] < 6.7) & (df_filtered['lon'] > 3.0)]
        
        fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="Neighborhood_Tier",
                                    size="Bedrooms", hover_name="Location",
                                    zoom=10, mapbox_style="carto-positron", height=500,
                                    title="Geographic Distribution")
        st.plotly_chart(fig_map, use_container_width=True)

# ==========================================
# PAGE 2: AI PREDICTOR (Fast Inference)
# ==========================================
elif page == "🤖 AI Price Predictor":
    st.title("🤖 AI Property Valuator")
    st.markdown("Configure a property to estimate its market value.")
    
    if prod_model is not None:
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            # Smart Location Selector
            if not df.empty:
                loc_stats = df.groupby('Location')[['Neighborhood_Tier', 'dist_to_lekki', 'dist_to_ikeja', 'Is_Island']].mean()
                selected_loc = st.selectbox("Location", loc_stats.index.sort_values())
                loc_data = loc_stats.loc[selected_loc]
            else:
                # Fallback if data missing
                loc_data = pd.Series({'Neighborhood_Tier':3, 'dist_to_lekki':10, 'dist_to_ikeja':20, 'Is_Island':1})
            
            with c2: 
                beds = st.number_input("Bedrooms", 1, 10, 4)
                baths = st.number_input("Bathrooms", 1, 10, 4)
            with c3: 
                p_type = st.selectbox("Type", ['Detached Duplex', 'Semi-Detached', 'Terrace', 'Flat', 'Bungalow'])
            
            c4, c5, c6 = st.columns(3)
            with c4: pool = st.checkbox("Swimming Pool")
            with c5: new = st.checkbox("Newly Built", value=True)
            with c6: estate = st.checkbox("In Estate", value=True)
            
            submitted = st.form_submit_button("🚀 Predict Price")
            
        if submitted:
            # Build Input Dictionary
            input_data = {
                'Bedrooms': beds, 'Bathrooms': baths,
                'dist_to_lekki': loc_data['dist_to_lekki'],
                'dist_to_ikeja': loc_data['dist_to_ikeja'],
                'Is_Island': loc_data['Is_Island'],
                'Has_Pool': 1 if pool else 0, 'Has_BQ': 1, 'New_Build': 1 if new else 0,
                'Bathroom_per_Bedroom': baths/beds, 'In_Estate': 1 if estate else 0,
                'Neighborhood_Tier': loc_data['Neighborhood_Tier'],
                'Island_x_Beds': loc_data['Is_Island'] * beds,
                'Luxury_Score': (1 if pool else 0) + (1 if estate else 0) + (1 if new else 0)
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            # One-Hot Encoding
            input_df = pd.get_dummies(input_df)
            
            # Align with Training Columns (Fill missing columns with 0)
            for col in prod_features:
                if col not in input_df.columns: input_df[col] = 0
            
            # Handle Property Type manually
            target_col = f"Prop_Type_{p_type}"
            if target_col in input_df.columns: input_df[target_col] = 1
            
            # Select final columns in correct order
            input_df = input_df[prod_features]
            
            # Predict (Log Scale -> Naira)
            pred_log = prod_model.predict(input_df)[0]
            pred_naira = np.expm1(pred_log)
            
            st.markdown("---")
            st.success("✅ Valuation Complete")
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1: st.metric("Estimated Market Value", f"₦{pred_naira:,.0f}")
            with col_res2: st.info(f"**Tier:** {int(loc_data['Neighborhood_Tier'])}/5\n\n**Dist. to Lekki:** {loc_data['dist_to_lekki']:.1f}km")
    else:
        st.error("⚠️ Model not found. Please upload 'lagos_model.pkl' and 'model_features.pkl'.")