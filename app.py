import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Styling */
    .reportview-container {
        background: #0F172A;
    }
    
    /* Gradient headers */
    .gradient-title {
        background: linear-gradient(90deg, #3B82F6 0%, #EC4899 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .gradient-subtitle {
        color: #94A3B8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Segment badges */
    .badge-high-value {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    .badge-regular {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    .badge-occasional {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    .badge-at-risk {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 15px;
    }
    
    /* Product recommendation styling */
    .prod-card {
        background: #1E293B;
        border-left: 5px solid #8B5CF6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts
@st.cache_resource
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'shopper_spectrum_models.pkl')
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
    except Exception:
        return None
    return artifacts

artifacts = load_models()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=120)
    st.markdown("### **Shopper Spectrum**")
    st.markdown("Customer Segmentation & Product Recommendations using Unsupervised & Supervised Machine Learning.")
    st.markdown("---")
    
    page = st.radio(
        "Navigate Modules",
        ["🏠 Home", "🎯 Customer Segmentation", "🛍 Product Recommendation", "📊 Business Dashboard"]
    )
    st.markdown("---")
    st.markdown("**Domain:** Retail & E-Commerce")
    st.markdown("**Techniques:** RFM Analysis, K-Means Clustering, Random Forest Classifier, Cosine Similarity Collaborative Filtering")

# Page 1: Home Page
if page == "🏠 Home":
    st.markdown('<div class="gradient-title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-subtitle">Customer Segmentation & Product Recommendation System</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>📣 Project Overview</h3>
            <p>The global e-commerce industry generates massive transaction datasets. Extracting value from this data requires mapping customer behaviors and serving personalized product recommendations. </p>
            <p><strong>Shopper Spectrum</strong> achieves this in a two-stage machine learning process:</p>
            <ul>
                <li><strong>Unsupervised Clustering:</strong> Segmenting customers into VIP, steady, and inactive segments based on Recency, Frequency, and Monetary (RFM) transaction metrics.</li>
                <li><strong>Collaborative Filtering:</strong> Matching products based on co-purchase history vector similarities to drive cross-selling.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🧠 Target Segments Identified</h3>
            <ul>
                <li><span style="color: #10B981; font-weight: bold;">High-Value (VIP):</span> Recent, very frequent buyers spending significant amounts. Strategy: Early previews & rewards.</li>
                <li><span style="color: #3B82F6; font-weight: bold;">Regular:</span> Active shoppers with moderate spend. Strategy: Recommendation bundles.</li>
                <li><span style="color: #F59E0B; font-weight: bold;">Occasional:</span> Infrequent, low monetary spenders. Strategy: Seasonal discounts.</li>
                <li><span style="color: #EF4444; font-weight: bold;">At-Risk:</span> Inactive for a long period, rare purchasers. Strategy: Win-back offers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Page 2: Customer Segmentation Module
elif page == "🎯 Customer Segmentation":
    st.markdown('<div class="gradient-title">🎯 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-subtitle">Predict customer segment using trained Random Forest Classifier</div>', unsafe_allow_html=True)
    
    if artifacts is None:
        st.error("Model artifacts not found! Please run the notebook first to generate shopper_spectrum_models.pkl.")
    else:
        scaler = artifacts.get('scaler')
        model = artifacts.get('model')
        mapping = artifacts.get('cluster_mapping')
        if scaler is None or model is None or mapping is None:
            st.error("The model artifact file is missing required components. Recreate shopper_spectrum_models.pkl.")
            st.stop()
        
        st.markdown("""
        <div class="glass-card">
            <h4>Enter Customer Transaction Profile Details:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Grid input layout
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (Days since last purchase)", min_value=1, max_value=375, value=30, help="How many days ago did this customer buy from us?")
        with col2:
            frequency = st.number_input("Frequency (Total Invoices)", min_value=1, max_value=250, value=5, help="Number of distinct transactions this customer made.")
        with col3:
            monetary = st.number_input("Monetary Spend (GBP £)", min_value=1.0, max_value=300000.0, value=500.0, step=10.0, help="Total amount spent by this customer.")
            
        if st.button("Predict Customer Segment", use_container_width=True):
            # Preprocess inputs
            raw_input = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            log_input = np.log1p(raw_input)
            scaled_input = scaler.transform(log_input)
            
            # Predict
            pred_class = model.predict(scaled_input)[0]
            segment_label = mapping[pred_class]
            
            # Show Segment Badge and Specific Actions
            st.markdown("---")
            st.subheader("Prediction Result:")
            
            if segment_label == "High-Value":
                st.markdown('<div class="badge-high-value">✨ High-Value (VIP) Segment</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="glass-card" style="border-left: 5px solid #10B981;">
                    <h5>Segment Characteristics:</h5>
                    <p>These are your top customers. They shop very frequently, make recent purchases, and spend significant capital.</p>
                    <h5>Marketing Recommendations:</h5>
                    <ul>
                        <li>Invite to exclusive VIP loyalty program and points reward system.</li>
                        <li>Provide early access to new seasonal product launches.</li>
                        <li>Personalized communication with dedicated customer service reps.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif segment_label == "Regular":
                st.markdown('<div class="badge-regular">🔹 Regular Segment</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="glass-card" style="border-left: 5px solid #3B82F6;">
                    <h5>Segment Characteristics:</h5>
                    <p>Reliable shoppers. They buy moderately often and spend reasonable amounts. They are active.</p>
                    <h5>Marketing Recommendations:</h5>
                    <ul>
                        <li>Cross-sell premium items using customized recommendations.</li>
                        <li>Offer free shipping thresholds to increase basket size.</li>
                        <li>Send targeted newsletters featuring popular products.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif segment_label == "Occasional":
                st.markdown('<div class="badge-occasional">🔸 Occasional Segment</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="glass-card" style="border-left: 5px solid #F59E0B;">
                    <h5>Segment Characteristics:</h5>
                    <p>These customers buy occasionally and spend lower amounts. They are active but require prompts.</p>
                    <h5>Marketing Recommendations:</h5>
                    <ul>
                        <li>Promote limited-time coupons or weekend flash sales.</li>
                        <li>Design seasonal holiday marketing campaigns.</li>
                        <li>Suggest trending products via email templates.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else: # At-Risk
                st.markdown('<div class="badge-at-risk">⚠️ At-Risk Segment</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="glass-card" style="border-left: 5px solid #EF4444;">
                    <h5>Segment Characteristics:</h5>
                    <p>These customers haven't purchased in a long time. They have low loyalty metrics and are likely churning.</p>
                    <h5>Marketing Recommendations:</h5>
                    <ul>
                        <li>Send 'We Miss You' customized win-back emails.</li>
                        <li>Provide steep discount codes (e.g. 25% off) to reactivate them.</li>
                        <li>Trigger customer feedback surveys to understand their inactivity.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# Page 3: Product Recommendation Module
elif page == "🛍 Product Recommendation":
    st.markdown('<div class="gradient-title">🛍 Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-subtitle">Get top 5 recommended products based on collaborative filtering</div>', unsafe_allow_html=True)
    
    if artifacts is None:
        st.error("Model artifacts not found! Please run the notebook first to generate shopper_spectrum_models.pkl.")
    else:
        similarity_df = artifacts.get('similarity_df')
        products = artifacts.get('products', [])
        if similarity_df is None or not isinstance(products, list) or len(products) == 0:
            st.error("The recommendation artifacts are incomplete. Recreate shopper_spectrum_models.pkl.")
            st.stop()
        
        st.markdown("""
        <div class="glass-card">
            <h4>Search Product Inventory:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        selected_prod = st.selectbox(
            "Select or Type a Product Name:",
            products,
            help="Choose a product to get similar item recommendations."
        )
        
        if st.button("Get Similar Recommendations", use_container_width=True):
            if selected_prod not in similarity_df.columns:
                st.error("Selected product is not available in the similarity index. Please choose another item.")
            else:
                st.markdown("---")
                st.markdown(f"#### Customers who bought **'{selected_prod}'** also bought:")
                
                # Fetch top recommendations
                similar_items = similarity_df[selected_prod].sort_values(ascending=False)[1:6]
            
            col1, col2 = st.columns([1, 4])
            with col2:
                for idx, (prod, score) in enumerate(similar_items.items(), 1):
                    st.markdown(f"""
                    <div class="prod-card">
                        <strong>Recommendation #{idx}:</strong> {prod}<br/>
                        <span style="color: #94A3B8; font-size: 0.85rem;">Cosine Similarity Match: {score:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

# Page 4: Business Dashboard & Analytics
elif page == "📊 Business Dashboard":
    st.markdown('<div class="gradient-title">📊 Business Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-subtitle">Overview of E-Commerce Transactional Metrics & Segments</div>', unsafe_allow_html=True)
    
    # Grid of Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Customer Accounts", value="4,338")
    with col2:
        st.metric(label="Total Unique Products", value="3,887")
    with col3:
        st.metric(label="Top Country Market", value="United Kingdom")
    with col4:
        st.metric(label="Peak Shopping Hours", value="12 PM - 2 PM")
        
    st.markdown("---")
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("##### Customer Segment Distribution")
        # Render a simple segment pie chart description
        segment_distribution = {
            "VIP / High-Value": 10.5,
            "Regular Shoppers": 28.3,
            "Occasional Spenders": 36.4,
            "At-Risk / Dormant": 24.8
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(segment_distribution.values(), labels=segment_distribution.keys(), autopct='%1.1f%%', 
               colors=['#10B981', '#3B82F6', '#F59E0B', '#EF4444'], startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax.axis('equal')
        fig.patch.set_facecolor('#0E1525')
        ax.set_facecolor('#0E1525')
        # Update text color for labels
        for text in ax.texts:
            text.set_color('white')
        st.pyplot(fig)
        
    with col_plot2:
        st.markdown("##### Hourly Sales Activity (Transaction Volume)")
        # Show an hourly chart
        hours = list(range(6, 21))
        invoices = [1, 15, 380, 1100, 1800, 2600, 3100, 3300, 2900, 2400, 1900, 1200, 400, 50, 5]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hours, invoices, marker='o', color='#EC4899', linewidth=2)
        ax.set_xlabel('Hour of Day', color='white')
        ax.set_ylabel('Number of Transactions', color='white')
        fig.patch.set_facecolor('#0E1525')
        ax.set_facecolor('#0E1525')
        ax.tick_params(colors='white')
        ax.grid(color='#1E293B', linestyle='--')
        st.pyplot(fig)
