"""Unit tests for location-aware recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.data_loader import LocationAwareDataLoader
from src.models import (
    PopularityRecommender,
    LocationAwareCollaborativeFiltering,
    ContentBasedRecommender,
    HybridRecommender
)
from src.evaluation import RecommendationEvaluator


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data": {
            "interactions_file": "data/interactions.csv",
            "items_file": "data/items.csv",
            "users_file": "data/users.csv",
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42
        },
        "models": {
            "location_cf": {
                "n_factors": 10,
                "regularization": 0.01,
                "iterations": 10,
                "alpha": 1.0
            },
            "content_based": {
                "tfidf_max_features": 100,
                "location_weight": 0.3,
                "content_weight": 0.7
            },
            "hybrid": {
                "cf_weight": 0.6,
                "content_weight": 0.4
            }
        },
        "evaluation": {
            "metrics": ["precision@5", "recall@5"],
            "k_values": [5, 10]
        },
        "location": {
            "max_distance_km": 50.0,
            "distance_decay_factor": 0.1
        }
    }


@pytest.fixture
def sample_interactions():
    """Sample interactions DataFrame."""
    return pd.DataFrame({
        "user_id": ["user_1", "user_1", "user_2", "user_2", "user_3"],
        "item_id": ["item_1", "item_2", "item_1", "item_3", "item_2"],
        "rating": [5, 4, 3, 5, 2],
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
        "user_lat": [40.0, 40.0, 41.0, 41.0, 42.0],
        "user_lon": [-74.0, -74.0, -75.0, -75.0, -76.0],
        "item_lat": [40.1, 40.2, 40.1, 40.3, 40.2],
        "item_lon": [-74.1, -74.2, -74.1, -74.3, -74.2]
    })


@pytest.fixture
def sample_items():
    """Sample items DataFrame."""
    return pd.DataFrame({
        "item_id": ["item_1", "item_2", "item_3"],
        "title": ["Restaurant A", "Store B", "Attraction C"],
        "category": ["restaurant", "store", "attraction"],
        "latitude": [40.1, 40.2, 40.3],
        "longitude": [-74.1, -74.2, -74.3],
        "description": ["Great food", "Good prices", "Fun place"],
        "tags": ["food,local", "shopping,retail", "entertainment,tourist"]
    })


@pytest.fixture
def sample_users():
    """Sample users DataFrame."""
    return pd.DataFrame({
        "user_id": ["user_1", "user_2", "user_3"],
        "age": [25, 30, 35],
        "gender": ["M", "F", "M"],
        "latitude": [40.0, 41.0, 42.0],
        "longitude": [-74.0, -75.0, -76.0],
        "city": ["NYC", "Boston", "DC"]
    })


class TestLocationAwareDataLoader:
    """Test cases for LocationAwareDataLoader."""
    
    def test_init(self, sample_config):
        """Test data loader initialization."""
        loader = LocationAwareDataLoader(sample_config)
        assert loader.config == sample_config
        assert loader.interactions_df is None
        assert loader.items_df is None
        assert loader.users_df is None
    
    def test_calculate_distance_matrix(self, sample_config, sample_users, sample_items):
        """Test distance matrix calculation."""
        loader = LocationAwareDataLoader(sample_config)
        
        # Mock the geodesic function to return predictable distances
        with patch('src.data_loader.geodesic') as mock_geodesic:
            mock_geodesic.return_value.km = 10.0  # Fixed distance for testing
            
            distances = loader.calculate_distance_matrix(sample_users, sample_items)
            
            assert distances.shape == (len(sample_users), len(sample_items))
            assert np.all(distances == 10.0)
    
    def test_split_data(self, sample_config, sample_interactions):
        """Test data splitting."""
        loader = LocationAwareDataLoader(sample_config)
        
        train_df, val_df, test_df = loader.split_data(sample_interactions)
        
        assert len(train_df) + len(val_df) + len(test_df) == len(sample_interactions)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
    
    def test_get_user_item_matrix(self, sample_config, sample_interactions):
        """Test user-item matrix creation."""
        loader = LocationAwareDataLoader(sample_config)
        
        matrix, user_ids, item_ids = loader.get_user_item_matrix(sample_interactions)
        
        assert matrix.shape == (3, 3)  # 3 users, 3 items
        assert len(user_ids) == 3
        assert len(item_ids) == 3
        assert "user_1" in user_ids
        assert "item_1" in item_ids


class TestPopularityRecommender:
    """Test cases for PopularityRecommender."""
    
    def test_init(self, sample_config):
        """Test popularity recommender initialization."""
        recommender = PopularityRecommender(sample_config)
        assert recommender.config == sample_config
        assert not recommender.is_fitted
    
    def test_fit(self, sample_config, sample_interactions, sample_items):
        """Test popularity recommender fitting."""
        recommender = PopularityRecommender(sample_config)
        recommender.fit(sample_interactions, sample_items)
        
        assert recommender.is_fitted
        assert len(recommender.item_popularity) > 0
        assert len(recommender.popular_items) > 0
    
    def test_recommend(self, sample_config, sample_interactions, sample_items):
        """Test popularity recommendations."""
        recommender = PopularityRecommender(sample_config)
        recommender.fit(sample_interactions, sample_items)
        
        recommendations = recommender.recommend("user_1", n_recommendations=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)
        assert all(isinstance(score, (int, float)) for _, score in recommendations)
    
    def test_get_explanations(self, sample_config, sample_interactions, sample_items):
        """Test popularity explanations."""
        recommender = PopularityRecommender(sample_config)
        recommender.fit(sample_interactions, sample_items)
        
        explanation = recommender.get_explanations("user_1", "item_1")
        
        assert "explanation" in explanation
        assert "popularity_score" in explanation
        assert "interaction_count" in explanation


class TestLocationAwareCollaborativeFiltering:
    """Test cases for LocationAwareCollaborativeFiltering."""
    
    def test_init(self, sample_config):
        """Test location-aware CF initialization."""
        recommender = LocationAwareCollaborativeFiltering(sample_config)
        assert recommender.config == sample_config
        assert not recommender.is_fitted
        assert recommender.model is not None
    
    def test_fit(self, sample_config, sample_interactions, sample_items, sample_users):
        """Test location-aware CF fitting."""
        recommender = LocationAwareCollaborativeFiltering(sample_config)
        recommender.fit(sample_interactions, sample_items, sample_users)
        
        assert recommender.is_fitted
        assert len(recommender.user_id_to_idx) > 0
        assert len(recommender.item_id_to_idx) > 0
    
    def test_recommend(self, sample_config, sample_interactions, sample_items, sample_users):
        """Test location-aware CF recommendations."""
        recommender = LocationAwareCollaborativeFiltering(sample_config)
        recommender.fit(sample_interactions, sample_items, sample_users)
        
        recommendations = recommender.recommend("user_1", n_recommendations=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)
        assert all(isinstance(score, (int, float)) for _, score in recommendations)


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def test_init(self, sample_config):
        """Test content-based recommender initialization."""
        recommender = ContentBasedRecommender(sample_config)
        assert recommender.config == sample_config
        assert not recommender.is_fitted
        assert recommender.tfidf_vectorizer is not None
    
    def test_fit(self, sample_config, sample_interactions, sample_items):
        """Test content-based recommender fitting."""
        recommender = ContentBasedRecommender(sample_config)
        recommender.fit(sample_interactions, sample_items)
        
        assert recommender.is_fitted
        assert recommender.item_features is not None
        assert len(recommender.user_profiles) > 0
    
    def test_recommend(self, sample_config, sample_interactions, sample_items):
        """Test content-based recommendations."""
        recommender = ContentBasedRecommender(sample_config)
        recommender.fit(sample_interactions, sample_items)
        
        recommendations = recommender.recommend("user_1", n_recommendations=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)
        assert all(isinstance(score, (int, float)) for _, score in recommendations)


class TestRecommendationEvaluator:
    """Test cases for RecommendationEvaluator."""
    
    def test_init(self, sample_config):
        """Test evaluator initialization."""
        evaluator = RecommendationEvaluator(sample_config)
        assert evaluator.config == sample_config
        assert evaluator.metrics == sample_config["evaluation"]["metrics"]
        assert evaluator.k_values == sample_config["evaluation"]["k_values"]
    
    def test_precision_at_k(self, sample_config):
        """Test precision@k calculation."""
        evaluator = RecommendationEvaluator(sample_config)
        
        recommended_items = ["item_1", "item_2", "item_3"]
        relevant_items = {"item_1", "item_3"}
        
        precision = evaluator._precision_at_k(recommended_items, relevant_items, 3)
        assert precision == 2/3  # 2 relevant out of 3 recommended
    
    def test_recall_at_k(self, sample_config):
        """Test recall@k calculation."""
        evaluator = RecommendationEvaluator(sample_config)
        
        recommended_items = ["item_1", "item_2", "item_3"]
        relevant_items = {"item_1", "item_3", "item_4"}
        
        recall = evaluator._recall_at_k(recommended_items, relevant_items, 3)
        assert recall == 2/3  # 2 relevant found out of 3 total relevant
    
    def test_hit_rate_at_k(self, sample_config):
        """Test hit rate@k calculation."""
        evaluator = RecommendationEvaluator(sample_config)
        
        recommended_items = ["item_1", "item_2", "item_3"]
        relevant_items = {"item_1", "item_4"}
        
        hit_rate = evaluator._hit_rate_at_k(recommended_items, relevant_items)
        assert hit_rate == 1.0  # At least one relevant item found
        
        # Test case with no hits
        relevant_items_no_hit = {"item_4", "item_5"}
        hit_rate_no_hit = evaluator._hit_rate_at_k(recommended_items, relevant_items_no_hit)
        assert hit_rate_no_hit == 0.0  # No relevant items found


if __name__ == "__main__":
    pytest.main([__file__])
