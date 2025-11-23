"""Evaluation metrics and utilities for recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluator for recommendation system metrics."""
    
    def __init__(self, config: Dict) -> None:
        """Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary containing evaluation parameters.
        """
        self.config = config
        self.metrics = config["evaluation"]["metrics"]
        self.k_values = config["evaluation"]["k_values"]
        
    def evaluate_model(self, 
                      model, 
                      test_df: pd.DataFrame, 
                      interactions_df: pd.DataFrame,
                      items_df: pd.DataFrame,
                      users_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Evaluate a recommendation model on test data.
        
        Args:
            model: Trained recommendation model.
            test_df: Test interactions DataFrame.
            interactions_df: Full interactions DataFrame for context.
            items_df: Items DataFrame.
            users_df: Optional users DataFrame.
            
        Returns:
            Dictionary of metric scores.
        """
        logger.info(f"Evaluating model: {model.__class__.__name__}")
        
        results = {}
        
        # Get unique users from test set
        test_users = test_df["user_id"].unique()
        
        # Calculate metrics for each k value
        for k in self.k_values:
            logger.info(f"Calculating metrics for k={k}")
            
            # Calculate metrics
            precision_scores = []
            recall_scores = []
            map_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            
            for user_id in test_users:
                # Get user's test items
                user_test_items = set(test_df[test_df["user_id"] == user_id]["item_id"].tolist())
                
                if len(user_test_items) == 0:
                    continue
                
                # Get recommendations
                try:
                    recommendations = model.recommend(user_id, n_recommendations=k, exclude_seen=True)
                    recommended_items = [item_id for item_id, _ in recommendations]
                except Exception as e:
                    logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                    continue
                
                # Calculate metrics
                precision = self._precision_at_k(recommended_items, user_test_items, k)
                recall = self._recall_at_k(recommended_items, user_test_items, len(user_test_items))
                map_score = self._map_at_k(recommended_items, user_test_items, k)
                ndcg = self._ndcg_at_k(recommended_items, user_test_items, k)
                hit_rate = self._hit_rate_at_k(recommended_items, user_test_items)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                map_scores.append(map_score)
                ndcg_scores.append(ndcg)
                hit_rate_scores.append(hit_rate)
            
            # Average metrics across users
            results[f"precision@{k}"] = np.mean(precision_scores) if precision_scores else 0.0
            results[f"recall@{k}"] = np.mean(recall_scores) if recall_scores else 0.0
            results[f"map@{k}"] = np.mean(map_scores) if map_scores else 0.0
            results[f"ndcg@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            results[f"hit_rate@{k}"] = np.mean(hit_rate_scores) if hit_rate_scores else 0.0
        
        logger.info(f"Evaluation completed. Results: {results}")
        return results
    
    def _precision_at_k(self, recommended_items: List[str], relevant_items: set, k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & relevant_items)
        
        return relevant_recommended / k
    
    def _recall_at_k(self, recommended_items: List[str], relevant_items: set, total_relevant: int) -> float:
        """Calculate Recall@K."""
        if total_relevant == 0:
            return 0.0
        
        recommended_set = set(recommended_items)
        relevant_recommended = len(recommended_set & relevant_items)
        
        return relevant_recommended / total_relevant
    
    def _map_at_k(self, recommended_items: List[str], relevant_items: set, k: int) -> float:
        """Calculate Mean Average Precision@K."""
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def _ndcg_at_k(self, recommended_items: List[str], relevant_items: set, k: int) -> float:
        """Calculate NDCG@K."""
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        
        # Create relevance scores (1 for relevant, 0 for not relevant)
        relevance_scores = [1 if item in relevant_items else 0 for item in recommended_k]
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_relevance):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _hit_rate_at_k(self, recommended_items: List[str], relevant_items: set) -> float:
        """Calculate Hit Rate@K."""
        recommended_set = set(recommended_items)
        return 1.0 if len(recommended_set & relevant_items) > 0 else 0.0
    
    def calculate_coverage(self, model, interactions_df: pd.DataFrame, items_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate catalog coverage metrics.
        
        Args:
            model: Trained recommendation model.
            interactions_df: Interactions DataFrame.
            items_df: Items DataFrame.
            
        Returns:
            Dictionary of coverage metrics.
        """
        logger.info("Calculating coverage metrics...")
        
        # Get all users
        users = interactions_df["user_id"].unique()
        all_recommended_items = set()
        
        # Get recommendations for all users
        for user_id in users:
            try:
                recommendations = model.recommend(user_id, n_recommendations=20, exclude_seen=True)
                recommended_items = [item_id for item_id, _ in recommendations]
                all_recommended_items.update(recommended_items)
            except Exception as e:
                logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                continue
        
        # Calculate coverage
        total_items = len(items_df)
        catalog_coverage = len(all_recommended_items) / total_items if total_items > 0 else 0.0
        
        return {
            "catalog_coverage": catalog_coverage,
            "unique_items_recommended": len(all_recommended_items),
            "total_items": total_items
        }
    
    def calculate_diversity(self, model, interactions_df: pd.DataFrame, items_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate diversity metrics.
        
        Args:
            model: Trained recommendation model.
            interactions_df: Interactions DataFrame.
            items_df: Items DataFrame.
            
        Returns:
            Dictionary of diversity metrics.
        """
        logger.info("Calculating diversity metrics...")
        
        users = interactions_df["user_id"].unique()[:100]  # Sample for efficiency
        intra_list_diversities = []
        
        for user_id in users:
            try:
                recommendations = model.recommend(user_id, n_recommendations=10, exclude_seen=True)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                if len(recommended_items) < 2:
                    continue
                
                # Calculate intra-list diversity (simplified version)
                # In practice, you'd use item features or categories
                diversity = len(set(recommended_items)) / len(recommended_items)
                intra_list_diversities.append(diversity)
                
            except Exception as e:
                logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                continue
        
        return {
            "intra_list_diversity": np.mean(intra_list_diversities) if intra_list_diversities else 0.0,
            "users_evaluated": len(intra_list_diversities)
        }
    
    def create_leaderboard(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a model comparison leaderboard.
        
        Args:
            model_results: Dictionary mapping model names to their evaluation results.
            
        Returns:
            DataFrame with model comparison results.
        """
        logger.info("Creating model leaderboard...")
        
        # Flatten results for DataFrame
        leaderboard_data = []
        
        for model_name, results in model_results.items():
            row = {"model": model_name}
            row.update(results)
            leaderboard_data.append(row)
        
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by primary metric (NDCG@10)
        if "ndcg@10" in leaderboard_df.columns:
            leaderboard_df = leaderboard_df.sort_values("ndcg@10", ascending=False)
        
        return leaderboard_df
