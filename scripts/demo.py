"""Streamlit demo for location-aware recommendation system."""

import logging
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from geopy.distance import geodesic

from src.data_loader import LocationAwareDataLoader
from src.models import (
    PopularityRecommender,
    LocationAwareCollaborativeFiltering,
    ContentBasedRecommender,
    HybridRecommender
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Location-Aware Recommendations",
    page_icon="📍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_and_models():
    """Load data and trained models."""
    try:
        # Load configuration
        config_path = Path("configs/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load data
        data_loader = LocationAwareDataLoader(config)
        interactions_df, items_df, users_df = data_loader.load_data()
        
        # Load trained models (if available)
        models = {}
        model_paths = {
            "Popularity": "models/popularity_model.pkl",
            "Location-Aware CF": "models/location_cf_model.pkl",
            "Content-Based": "models/content_model.pkl",
            "Hybrid": "models/hybrid_model.pkl"
        }
        
        for model_name, model_path in model_paths.items():
            if Path(model_path).exists():
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            else:
                # Train model on the fly if not saved
                logger.info(f"Training {model_name} on the fly...")
                if model_name == "Popularity":
                    model = PopularityRecommender(config)
                elif model_name == "Location-Aware CF":
                    model = LocationAwareCollaborativeFiltering(config)
                elif model_name == "Content-Based":
                    model = ContentBasedRecommender(config)
                elif model_name == "Hybrid":
                    model = HybridRecommender(config)
                
                model.fit(interactions_df, items_df, users_df)
                models[model_name] = model
        
        return config, interactions_df, items_df, users_df, models
    
    except Exception as e:
        st.error(f"Error loading data and models: {e}")
        return None, None, None, None, {}


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates."""
    return geodesic((lat1, lon1), (lat2, lon2)).km


def format_distance(distance: float) -> str:
    """Format distance for display."""
    if distance < 1:
        return f"{distance * 1000:.0f}m"
    else:
        return f"{distance:.1f}km"


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">📍 Location-Aware Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data and models
    config, interactions_df, items_df, users_df, models = load_data_and_models()
    
    if config is None:
        st.error("Failed to load configuration. Please check the setup.")
        return
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Recommendation Model",
        list(models.keys()),
        index=0
    )
    
    # User selection
    if users_df is not None:
        user_options = users_df["user_id"].tolist()
        selected_user = st.sidebar.selectbox(
            "Select User",
            user_options,
            index=0
        )
        
        # Get user location
        user_row = users_df[users_df["user_id"] == selected_user].iloc[0]
        user_lat = user_row["latitude"]
        user_lon = user_row["longitude"]
        
        st.sidebar.info(f"User Location: {user_lat:.4f}, {user_lon:.4f}")
    else:
        st.sidebar.error("No user data available")
        return
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Recommendations")
        
        if selected_model in models:
            model = models[selected_model]
            
            try:
                # Get recommendations
                recommendations = model.recommend(
                    selected_user, 
                    n_recommendations=n_recommendations,
                    exclude_seen=True
                )
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations using {selected_model}")
                    
                    # Display recommendations
                    for i, (item_id, score) in enumerate(recommendations, 1):
                        # Get item details
                        item_row = items_df[items_df["item_id"] == item_id].iloc[0]
                        
                        # Calculate distance
                        distance = calculate_distance(
                            user_lat, user_lon,
                            item_row["latitude"], item_row["longitude"]
                        )
                        
                        # Display recommendation
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-item">
                                <h4>#{i} {item_row['title']}</h4>
                                <p><strong>Category:</strong> {item_row['category']}</p>
                                <p><strong>Distance:</strong> {format_distance(distance)}</p>
                                <p><strong>Score:</strong> {score:.3f}</p>
                                <p><strong>Description:</strong> {item_row['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show explanation
                            if st.button(f"Explain Recommendation #{i}", key=f"explain_{i}"):
                                explanation = model.get_explanations(selected_user, item_id)
                                st.info(f"**Explanation:** {explanation.get('explanation', 'No explanation available')}")
                                
                                # Show additional details
                                for key, value in explanation.items():
                                    if key != "explanation" and value is not None:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.warning("No recommendations found for this user.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
        else:
            st.error(f"Model {selected_model} not available.")
    
    with col2:
        st.header("User Statistics")
        
        # User interaction history
        user_interactions = interactions_df[interactions_df["user_id"] == selected_user]
        
        if not user_interactions.empty:
            st.metric("Total Interactions", len(user_interactions))
            st.metric("Average Rating", f"{user_interactions['rating'].mean():.2f}")
            
            # Rating distribution
            rating_counts = user_interactions["rating"].value_counts().sort_index()
            st.bar_chart(rating_counts)
            
            # Recent interactions
            st.subheader("Recent Interactions")
            recent_interactions = user_interactions.nlargest(5, "timestamp")
            
            for _, interaction in recent_interactions.iterrows():
                item_row = items_df[items_df["item_id"] == interaction["item_id"]].iloc[0]
                distance = calculate_distance(
                    user_lat, user_lon,
                    item_row["latitude"], item_row["longitude"]
                )
                
                st.write(f"**{item_row['title']}** - Rating: {interaction['rating']} - Distance: {format_distance(distance)}")
        else:
            st.info("No interaction history for this user.")
        
        # Model comparison
        st.header("Model Comparison")
        
        if len(models) > 1:
            comparison_data = []
            
            for model_name, model in models.items():
                try:
                    recommendations = model.recommend(selected_user, n_recommendations=5, exclude_seen=True)
                    comparison_data.append({
                        "Model": model_name,
                        "Recommendations": len(recommendations),
                        "Avg Score": np.mean([score for _, score in recommendations]) if recommendations else 0
                    })
                except:
                    comparison_data.append({
                        "Model": model_name,
                        "Recommendations": 0,
                        "Avg Score": 0
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Location-Aware Recommendation System Demo
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
