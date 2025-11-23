# Location-Aware Recommendation System

A production-ready location-aware recommendation system that leverages geographical context to provide personalized recommendations based on user proximity and preferences.

## Features

- **Multiple Recommendation Models**: Popularity-based, Location-aware Collaborative Filtering, Content-based, and Hybrid approaches
- **Geographic Context**: Incorporates user and item locations using geodesic distance calculations
- **Comprehensive Evaluation**: Precision@K, Recall@K, MAP@K, NDCG@K, Hit Rate, Coverage, and Diversity metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production-Ready**: Clean code structure, type hints, comprehensive testing, and CI/CD setup

## Project Structure

```
├── src/                    # Source code modules
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── models.py          # Recommendation models
│   └── evaluation.py      # Evaluation metrics
├── configs/               # Configuration files
│   └── config.yaml       # Main configuration
├── data/                  # Data files (generated if not present)
├── models/                # Trained model artifacts
├── scripts/               # Training and demo scripts
│   ├── train.py          # Main training pipeline
│   └── demo.py           # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Results and visualizations
├── notebooks/             # Jupyter notebooks for exploration
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Location-Aware-Recommendation-System.git
cd Location-Aware-Recommendation-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Models

```bash
# Run the training pipeline
python scripts/train.py
```

This will:
- Generate synthetic location-aware data if no data files exist
- Train multiple recommendation models
- Evaluate models and create a leaderboard
- Save results to `assets/model_results.csv`

### 3. Interactive Demo

```bash
# Launch the Streamlit demo
streamlit run scripts/demo.py
```

The demo provides:
- User selection and recommendation generation
- Model comparison and explanations
- Interactive exploration of recommendations
- Distance calculations and location context

## Configuration

The system is configured via `configs/config.yaml`:

```yaml
# Data configuration
data:
  interactions_file: "data/interactions.csv"
  items_file: "data/items.csv"
  users_file: "data/users.csv"
  test_size: 0.2
  val_size: 0.1
  random_state: 42

# Model configuration
models:
  location_cf:
    n_factors: 50
    regularization: 0.01
    iterations: 50
    alpha: 1.0
  
  content_based:
    tfidf_max_features: 1000
    location_weight: 0.3
    content_weight: 0.7
  
  hybrid:
    cf_weight: 0.6
    content_weight: 0.4

# Evaluation configuration
evaluation:
  metrics: ["precision@5", "recall@5", "map@5", "ndcg@5", "hit_rate@5"]
  k_values: [5, 10, 20]
  cross_validation_folds: 5

# Location configuration
location:
  max_distance_km: 50.0
  distance_decay_factor: 0.1
  coordinate_precision: 6
```

## Data Schema

### Interactions (`interactions.csv`)
- `user_id`: Unique user identifier
- `item_id`: Unique item identifier
- `rating`: User rating (1-5 scale)
- `timestamp`: Interaction timestamp
- `user_lat`, `user_lon`: User location coordinates
- `item_lat`, `item_lon`: Item location coordinates

### Items (`items.csv`)
- `item_id`: Unique item identifier
- `title`: Item name/title
- `category`: Item category (restaurant, store, attraction, etc.)
- `latitude`, `longitude`: Item location coordinates
- `description`: Item description text
- `tags`: Comma-separated tags

### Users (`users.csv`)
- `user_id`: Unique user identifier
- `age`: User age
- `gender`: User gender
- `latitude`, `longitude`: User location coordinates
- `city`: User city

## Models

### 1. Popularity Recommender
Baseline model that recommends items based on overall popularity (average rating and interaction count).

### 2. Location-Aware Collaborative Filtering
Uses Alternating Least Squares (ALS) with geographic distance penalties to incorporate location context into collaborative filtering.

### 3. Content-Based Recommender
Leverages item descriptions and tags using TF-IDF vectorization, combined with location features.

### 4. Hybrid Recommender
Combines collaborative filtering and content-based approaches with configurable weights.

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **MAP@K**: Mean Average Precision considering ranking
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that can be recommended
- **Diversity**: Intra-list diversity of recommendations

## API Usage

```python
from src.data_loader import LocationAwareDataLoader
from src.models import LocationAwareCollaborativeFiltering
from src.evaluation import RecommendationEvaluator

# Load configuration
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
data_loader = LocationAwareDataLoader(config)
interactions_df, items_df, users_df = data_loader.load_data()

# Train model
model = LocationAwareCollaborativeFiltering(config)
model.fit(interactions_df, items_df, users_df)

# Generate recommendations
recommendations = model.recommend("user_123", n_recommendations=10)

# Get explanations
explanation = model.get_explanations("user_123", "item_456")
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Performance

The system is designed for scalability and performance:

- **Efficient Distance Calculations**: Uses geopy for accurate geodesic distance calculations
- **Sparse Matrix Operations**: Leverages scipy sparse matrices for memory efficiency
- **Caching**: Streamlit demo includes data and model caching
- **Parallel Processing**: Supports parallel evaluation across users

## Extending the System

### Adding New Models

1. Inherit from `BaseRecommender` in `src/models.py`
2. Implement `fit()` and `recommend()` methods
3. Add model configuration to `configs/config.yaml`
4. Update training script to include new model

### Adding New Metrics

1. Add metric calculation method to `RecommendationEvaluator`
2. Update configuration to include new metric
3. Add metric to evaluation pipeline

### Custom Data Sources

1. Modify `LocationAwareDataLoader` to support new data formats
2. Update data schema documentation
3. Add data validation and preprocessing steps

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Memory Issues**: Reduce `n_factors` in model configuration for large datasets
3. **Slow Performance**: Use smaller `k_values` in evaluation configuration
4. **No Recommendations**: Check if user exists in training data and has sufficient interactions

### Debug Mode

Enable debug logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Ensure code passes linting and tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with modern Python libraries: scikit-learn, pandas, numpy, implicit, geopy
- UI powered by Streamlit
- Testing with pytest
- Code quality with black, ruff, and mypy
# Location-Aware-Recommendation-System
