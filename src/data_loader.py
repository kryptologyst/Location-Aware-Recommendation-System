"""Data loading and preprocessing utilities for location-aware recommendations."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class LocationAwareDataLoader:
    """Data loader for location-aware recommendation systems."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the data loader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and parameters.
        """
        self.config = config
        self.interactions_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Load all data files.
        
        Returns:
            Tuple of (interactions_df, items_df, users_df).
        """
        logger.info("Loading data files...")
        
        # Load interactions
        interactions_path = Path(self.config["data"]["interactions_file"])
        if interactions_path.exists():
            self.interactions_df = pd.read_csv(interactions_path)
        else:
            logger.warning(f"Interactions file not found at {interactions_path}, generating synthetic data...")
            self.interactions_df = self._generate_synthetic_interactions()
            
        # Load items
        items_path = Path(self.config["data"]["items_file"])
        if items_path.exists():
            self.items_df = pd.read_csv(items_path)
        else:
            logger.warning(f"Items file not found at {items_path}, generating synthetic data...")
            self.items_df = self._generate_synthetic_items()
            
        # Load users (optional)
        users_path = Path(self.config["data"]["users_file"])
        if users_path.exists():
            self.users_df = pd.read_csv(users_path)
        else:
            logger.info("Users file not found, generating synthetic data...")
            self.users_df = self._generate_synthetic_users()
            
        logger.info(f"Loaded {len(self.interactions_df)} interactions, {len(self.items_df)} items, {len(self.users_df) if self.users_df is not None else 0} users")
        
        return self.interactions_df, self.items_df, self.users_df
    
    def _generate_synthetic_interactions(self) -> pd.DataFrame:
        """Generate synthetic interaction data with location-aware patterns."""
        np.random.seed(self.config["data"]["random_state"])
        random.seed(self.config["data"]["random_state"])
        
        n_users = 1000
        n_items = 200
        
        interactions = []
        
        # Generate interactions with location bias
        for user_id in range(n_users):
            # Simulate user location (random city coordinates)
            user_lat = np.random.uniform(25.0, 50.0)  # US latitude range
            user_lon = np.random.uniform(-125.0, -65.0)  # US longitude range
            
            # Generate items with location bias
            n_interactions = np.random.poisson(20)  # Average 20 interactions per user
            
            for _ in range(n_interactions):
                item_id = np.random.randint(0, n_items)
                
                # Simulate rating based on distance and popularity
                item_lat = np.random.uniform(25.0, 50.0)
                item_lon = np.random.uniform(-125.0, -65.0)
                
                distance = geodesic((user_lat, user_lon), (item_lat, item_lon)).km
                
                # Rating decreases with distance and has some randomness
                base_rating = 5 - (distance / 100)  # Distance penalty
                rating = max(1, min(5, int(base_rating + np.random.normal(0, 1))))
                
                # Add timestamp (last 2 years)
                timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 730))
                
                interactions.append({
                    "user_id": f"user_{user_id}",
                    "item_id": f"item_{item_id}",
                    "rating": rating,
                    "timestamp": timestamp,
                    "user_lat": user_lat,
                    "user_lon": user_lon,
                    "item_lat": item_lat,
                    "item_lon": item_lon
                })
        
        return pd.DataFrame(interactions)
    
    def _generate_synthetic_items(self) -> pd.DataFrame:
        """Generate synthetic item data with location information."""
        np.random.seed(self.config["data"]["random_state"])
        
        n_items = 200
        categories = ["restaurant", "store", "attraction", "hotel", "service"]
        
        items = []
        for item_id in range(n_items):
            lat = np.random.uniform(25.0, 50.0)
            lon = np.random.uniform(-125.0, -65.0)
            category = np.random.choice(categories)
            
            items.append({
                "item_id": f"item_{item_id}",
                "title": f"{category.title()} {item_id}",
                "category": category,
                "latitude": lat,
                "longitude": lon,
                "description": f"A {category} located at coordinates ({lat:.4f}, {lon:.4f})",
                "tags": f"{category},local,business"
            })
        
        return pd.DataFrame(items)
    
    def _generate_synthetic_users(self) -> pd.DataFrame:
        """Generate synthetic user data."""
        np.random.seed(self.config["data"]["random_state"])
        
        n_users = 1000
        users = []
        
        for user_id in range(n_users):
            lat = np.random.uniform(25.0, 50.0)
            lon = np.random.uniform(-125.0, -65.0)
            
            users.append({
                "user_id": f"user_{user_id}",
                "age": np.random.randint(18, 80),
                "gender": np.random.choice(["M", "F", "Other"]),
                "latitude": lat,
                "longitude": lon,
                "city": f"City_{user_id % 50}"  # 50 different cities
            })
        
        return pd.DataFrame(users)
    
    def calculate_distance_matrix(self, user_locations: pd.DataFrame, item_locations: pd.DataFrame) -> np.ndarray:
        """Calculate distance matrix between users and items.
        
        Args:
            user_locations: DataFrame with user_id, latitude, longitude columns.
            item_locations: DataFrame with item_id, latitude, longitude columns.
            
        Returns:
            Distance matrix of shape (n_users, n_items).
        """
        distances = np.zeros((len(user_locations), len(item_locations)))
        
        for i, user_row in user_locations.iterrows():
            for j, item_row in item_locations.iterrows():
                user_coords = (user_row["latitude"], user_row["longitude"])
                item_coords = (item_row["latitude"], item_row["longitude"])
                distances[i, j] = geodesic(user_coords, item_coords).km
        
        return distances
    
    def split_data(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions into train, validation, and test sets.
        
        Args:
            interactions_df: DataFrame with interactions.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # Sort by timestamp for temporal splitting
        interactions_df = interactions_df.sort_values("timestamp")
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            interactions_df,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=interactions_df["user_id"]  # Stratify by user to ensure all users in train
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config["data"]["val_size"] / (1 - self.config["data"]["test_size"]),
            random_state=self.config["data"]["random_state"],
            stratify=train_val_df["user_id"]
        )
        
        logger.info(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return train_df, val_df, test_df
    
    def get_user_item_matrix(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """Convert interactions to user-item matrix.
        
        Args:
            interactions_df: DataFrame with interactions.
            
        Returns:
            Tuple of (matrix, user_ids, item_ids).
        """
        # Create pivot table
        matrix_df = interactions_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0
        )
        
        matrix = matrix_df.values
        user_ids = matrix_df.index.tolist()
        item_ids = matrix_df.columns.tolist()
        
        return matrix, user_ids, item_ids
