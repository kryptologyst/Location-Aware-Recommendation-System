#!/usr/bin/env python3
"""Main training script for location-aware recommendation system."""

import logging
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.data_loader import LocationAwareDataLoader
from src.models import (
    PopularityRecommender,
    LocationAwareCollaborativeFiltering,
    ContentBasedRecommender,
    HybridRecommender
)
from src.evaluation import RecommendationEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    """Main training pipeline."""
    logger.info("Starting location-aware recommendation system training...")
    
    # Load configuration
    config_path = Path("configs/config.yaml")
    config = load_config(config_path)
    
    # Set random seeds
    set_random_seeds(config["data"]["random_state"])
    
    # Load data
    logger.info("Loading data...")
    data_loader = LocationAwareDataLoader(config)
    interactions_df, items_df, users_df = data_loader.load_data()
    
    # Split data
    logger.info("Splitting data...")
    train_df, val_df, test_df = data_loader.split_data(interactions_df)
    
    # Initialize models
    models = {
        "Popularity": PopularityRecommender(config),
        "Location-Aware CF": LocationAwareCollaborativeFiltering(config),
        "Content-Based": ContentBasedRecommender(config),
        "Hybrid": HybridRecommender(config)
    }
    
    # Train models
    logger.info("Training models...")
    trained_models = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        try:
            model.fit(train_df, items_df, users_df)
            trained_models[model_name] = model
            logger.info(f"Successfully trained {model_name}")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = RecommendationEvaluator(config)
    model_results = {}
    
    for model_name, model in trained_models.items():
        logger.info(f"Evaluating {model_name}...")
        try:
            # Evaluate on test set
            results = evaluator.evaluate_model(model, test_df, interactions_df, items_df, users_df)
            
            # Add coverage and diversity metrics
            coverage_results = evaluator.calculate_coverage(model, interactions_df, items_df)
            diversity_results = evaluator.calculate_diversity(model, interactions_df, items_df)
            
            results.update(coverage_results)
            results.update(diversity_results)
            
            model_results[model_name] = results
            logger.info(f"Evaluation completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    # Create leaderboard
    if model_results:
        logger.info("Creating model leaderboard...")
        leaderboard_df = evaluator.create_leaderboard(model_results)
        
        # Save results
        results_path = Path("assets/model_results.csv")
        results_path.parent.mkdir(exist_ok=True)
        leaderboard_df.to_csv(results_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
        logger.info("Model Leaderboard:")
        logger.info(f"\n{leaderboard_df.to_string(index=False)}")
        
        # Save individual model results
        for model_name, results in model_results.items():
            logger.info(f"\n{model_name} Results:")
            for metric, value in results.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Training pipeline completed!")


if __name__ == "__main__":
    main()
