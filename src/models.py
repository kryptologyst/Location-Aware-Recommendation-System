"""Location-aware recommendation models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Base class for recommendation models."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the recommender with configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the recommendation model.
        
        Args:
            interactions_df: DataFrame with user-item interactions.
            items_df: DataFrame with item information.
            users_df: Optional DataFrame with user information.
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, n_recommendations: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
        """
        pass
    
    def get_explanations(self, user_id: str, item_id: str) -> Dict[str, Union[str, float]]:
        """Get explanation for why an item was recommended to a user.
        
        Args:
            user_id: User ID.
            item_id: Item ID.
            
        Returns:
            Dictionary with explanation details.
        """
        return {"explanation": "No explanation available for this model"}


class PopularityRecommender(BaseRecommender):
    """Popularity-based baseline recommender."""
    
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the popularity model."""
        logger.info("Fitting popularity recommender...")
        
        # Calculate item popularity (average rating)
        self.item_popularity = interactions_df.groupby("item_id")["rating"].mean().to_dict()
        self.item_counts = interactions_df.groupby("item_id").size().to_dict()
        
        # Sort by popularity
        self.popular_items = sorted(
            self.item_popularity.items(),
            key=lambda x: (x[1], self.item_counts[x[0]]),
            reverse=True
        )
        
        self.is_fitted = True
        logger.info(f"Fitted popularity model with {len(self.popular_items)} items")
    
    def recommend(self, user_id: str, n_recommendations: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generate popularity-based recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = []
        for item_id, score in self.popular_items[:n_recommendations]:
            recommendations.append((item_id, score))
        
        return recommendations
    
    def get_explanations(self, user_id: str, item_id: str) -> Dict[str, Union[str, float]]:
        """Get popularity-based explanation."""
        return {
            "explanation": f"This item is popular with an average rating of {self.item_popularity.get(item_id, 0):.2f}",
            "popularity_score": self.item_popularity.get(item_id, 0),
            "interaction_count": self.item_counts.get(item_id, 0)
        }


class LocationAwareCollaborativeFiltering(BaseRecommender):
    """Location-aware collaborative filtering using ALS."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the location-aware CF model."""
        super().__init__(config)
        self.model = AlternatingLeastSquares(
            factors=self.config["models"]["location_cf"]["n_factors"],
            regularization=self.config["models"]["location_cf"]["regularization"],
            iterations=self.config["models"]["location_cf"]["iterations"],
            alpha=self.config["models"]["location_cf"]["alpha"],
            random_state=42
        )
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
        self.distance_matrix = None
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the location-aware collaborative filtering model."""
        logger.info("Fitting location-aware collaborative filtering model...")
        
        # Create user and item mappings
        unique_users = interactions_df["user_id"].unique()
        unique_items = interactions_df["item_id"].unique()
        
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_idx.items()}
        
        # Create user-item matrix
        matrix = np.zeros((len(unique_users), len(unique_items)))
        for _, row in interactions_df.iterrows():
            user_idx = self.user_id_to_idx[row["user_id"]]
            item_idx = self.item_id_to_idx[row["item_id"]]
            matrix[user_idx, item_idx] = row["rating"]
        
        # Calculate distance matrix if user locations are available
        if users_df is not None and "latitude" in users_df.columns and "longitude" in users_df.columns:
            self._calculate_distance_matrix(users_df, items_df)
        
        # Fit ALS model
        self.model.fit(matrix)
        
        self.is_fitted = True
        logger.info(f"Fitted location-aware CF model with {len(unique_users)} users and {len(unique_items)} items")
    
    def _calculate_distance_matrix(self, users_df: pd.DataFrame, items_df: pd.DataFrame) -> None:
        """Calculate distance matrix between users and items."""
        logger.info("Calculating distance matrix...")
        
        n_users = len(self.user_id_to_idx)
        n_items = len(self.item_id_to_idx)
        self.distance_matrix = np.zeros((n_users, n_items))
        
        for user_id, user_idx in self.user_id_to_idx.items():
            user_row = users_df[users_df["user_id"] == user_id].iloc[0]
            user_coords = (user_row["latitude"], user_row["longitude"])
            
            for item_id, item_idx in self.item_id_to_idx.items():
                item_row = items_df[items_df["item_id"] == item_id].iloc[0]
                item_coords = (item_row["latitude"], item_row["longitude"])
                
                distance = geodesic(user_coords, item_coords).km
                self.distance_matrix[user_idx, item_idx] = distance
        
        logger.info("Distance matrix calculated")
    
    def recommend(self, user_id: str, n_recommendations: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generate location-aware recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_id_to_idx:
            logger.warning(f"User {user_id} not found in training data, returning popular items")
            return []
        
        user_idx = self.user_id_to_idx[user_id]
        
        # Get raw recommendations from ALS
        item_scores = self.model.user_factors[user_idx] @ self.model.item_factors.T
        
        # Apply location penalty if distance matrix is available
        if self.distance_matrix is not None:
            user_distances = self.distance_matrix[user_idx]
            max_distance = self.config["location"]["max_distance_km"]
            decay_factor = self.config["location"]["distance_decay_factor"]
            
            # Apply distance penalty
            location_penalty = np.exp(-decay_factor * user_distances / max_distance)
            item_scores *= location_penalty
        
        # Get top recommendations
        if exclude_seen:
            # Set seen items to very low score
            seen_items = set()
            for item_id, item_idx in self.item_id_to_idx.items():
                if self.model.user_factors[user_idx] @ self.model.item_factors[item_idx] > 0:
                    seen_items.add(item_idx)
            
            for item_idx in seen_items:
                item_scores[item_idx] = -np.inf
        
        # Get top items
        top_indices = np.argsort(item_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            if item_scores[idx] > -np.inf:  # Skip excluded items
                item_id = self.idx_to_item_id[idx]
                score = float(item_scores[idx])
                recommendations.append((item_id, score))
        
        return recommendations
    
    def get_explanations(self, user_id: str, item_id: str) -> Dict[str, Union[str, float]]:
        """Get explanation for location-aware CF recommendation."""
        if not self.is_fitted:
            return {"explanation": "Model not fitted"}
        
        if user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
            return {"explanation": "User or item not found in training data"}
        
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        # Calculate collaborative filtering score
        cf_score = float(self.model.user_factors[user_idx] @ self.model.item_factors[item_idx])
        
        explanation = f"Based on collaborative filtering, this item has a score of {cf_score:.3f}"
        
        if self.distance_matrix is not None:
            distance = self.distance_matrix[user_idx, item_idx]
            explanation += f" and is {distance:.1f} km away from you"
        
        return {
            "explanation": explanation,
            "cf_score": cf_score,
            "distance_km": float(self.distance_matrix[user_idx, item_idx]) if self.distance_matrix is not None else None
        }


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender with location features."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the content-based model."""
        super().__init__(config)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config["models"]["content_based"]["tfidf_max_features"],
            stop_words="english"
        )
        self.item_features = None
        self.user_profiles = {}
        self.location_weight = self.config["models"]["content_based"]["location_weight"]
        self.content_weight = self.config["models"]["content_based"]["content_weight"]
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the content-based model."""
        logger.info("Fitting content-based recommender...")
        
        # Create item features from text descriptions
        item_texts = items_df["description"].fillna("") + " " + items_df["tags"].fillna("")
        self.item_features = self.tfidf_vectorizer.fit_transform(item_texts)
        
        # Build user profiles
        for user_id in interactions_df["user_id"].unique():
            user_items = interactions_df[interactions_df["user_id"] == user_id]["item_id"].tolist()
            
            if user_items:
                # Get item indices
                item_indices = [items_df[items_df["item_id"] == item_id].index[0] for item_id in user_items if item_id in items_df["item_id"].values]
                
                if item_indices:
                    # Average item features for user profile
                    user_profile = np.mean(self.item_features[item_indices], axis=0)
                    self.user_profiles[user_id] = user_profile
        
        self.is_fitted = True
        logger.info(f"Fitted content-based model with {len(self.user_profiles)} user profiles")
    
    def recommend(self, user_id: str, n_recommendations: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generate content-based recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate content similarity scores
        content_scores = cosine_similarity(user_profile, self.item_features).flatten()
        
        # Apply location penalty if user location is available
        # This is a simplified version - in practice, you'd need user location data
        location_scores = np.ones(len(content_scores))  # Placeholder
        
        # Combine content and location scores
        combined_scores = (self.content_weight * content_scores + 
                          self.location_weight * location_scores)
        
        # Get top recommendations
        top_indices = np.argsort(combined_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            item_id = self.items_df.iloc[idx]["item_id"]
            score = float(combined_scores[idx])
            recommendations.append((item_id, score))
        
        return recommendations
    
    def get_explanations(self, user_id: str, item_id: str) -> Dict[str, Union[str, float]]:
        """Get content-based explanation."""
        if not self.is_fitted:
            return {"explanation": "Model not fitted"}
        
        if user_id not in self.user_profiles:
            return {"explanation": "User not found in training data"}
        
        # Find item index
        item_idx = self.items_df[self.items_df["item_id"] == item_id].index[0]
        
        # Calculate similarity
        user_profile = self.user_profiles[user_id]
        item_features = self.item_features[item_idx]
        similarity = float(cosine_similarity(user_profile, item_features)[0, 0])
        
        return {
            "explanation": f"This item matches your preferences with a similarity score of {similarity:.3f}",
            "content_similarity": similarity,
            "item_description": self.items_df.iloc[item_idx]["description"]
        }


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining multiple approaches."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the hybrid model."""
        super().__init__(config)
        self.cf_model = LocationAwareCollaborativeFiltering(config)
        self.content_model = ContentBasedRecommender(config)
        self.cf_weight = config["models"]["hybrid"]["cf_weight"]
        self.content_weight = config["models"]["hybrid"]["content_weight"]
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the hybrid model."""
        logger.info("Fitting hybrid recommender...")
        
        # Fit individual models
        self.cf_model.fit(interactions_df, items_df, users_df)
        self.content_model.fit(interactions_df, items_df, users_df)
        
        self.is_fitted = True
        logger.info("Fitted hybrid model")
    
    def recommend(self, user_id: str, n_recommendations: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generate hybrid recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        cf_recs = dict(self.cf_model.recommend(user_id, n_recommendations * 2, exclude_seen))
        content_recs = dict(self.content_model.recommend(user_id, n_recommendations * 2, exclude_seen))
        
        # Combine scores
        all_items = set(cf_recs.keys()) | set(content_recs.keys())
        combined_scores = {}
        
        for item_id in all_items:
            cf_score = cf_recs.get(item_id, 0)
            content_score = content_recs.get(item_id, 0)
            
            combined_score = (self.cf_weight * cf_score + 
                             self.content_weight * content_score)
            combined_scores[item_id] = combined_score
        
        # Sort and return top recommendations
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
    
    def get_explanations(self, user_id: str, item_id: str) -> Dict[str, Union[str, float]]:
        """Get hybrid explanation."""
        cf_explanation = self.cf_model.get_explanations(user_id, item_id)
        content_explanation = self.content_model.get_explanations(user_id, item_id)
        
        return {
            "explanation": f"Hybrid recommendation combining collaborative filtering and content-based approaches",
            "cf_explanation": cf_explanation["explanation"],
            "content_explanation": content_explanation["explanation"],
            "cf_weight": self.cf_weight,
            "content_weight": self.content_weight
        }
