import sys
from pathlib import Path

import joblib

# Add data directory to path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
sys.path.append(str(project_root / 'data'))

data_path = project_root / 'data' / "digital_diet_mental_health.csv"

from mental_health_processing import MentalHealthProcessor

# Example usage
if __name__ == "__main__":

    print("Starting MENTAL HEALTH data processing...")

    # Configuration
    config = {
        "csv_file_path": data_path,
        "column_to_drop": "user_id", # ["user_id", "age", "gender", "location_type"],
        "columns_to_restructure": ["gender", "location_type"], # "gender", "location_type"
        "columns_to_normalize": [], # ["sleep_quality", "mood_rating", "stress_level", "weekly_anxiety_score", "weekly_depression_score"],
        "mapping_columns_to_map": {"gender": {"Male": 1, "Female": 2, "Other": 0}, "location_type": {"Urban": 2, "Suburban": 0, "Rural": 1}}, # "gender": {"Male": 1, "Female": 2, "Other": 0}, "location_type": {"Urban": 2, "Suburban": 0, "Rural": 1}
        "columns_to_move": ["mood_rating", "stress_level", "weekly_anxiety_score", "weekly_depression_sccore", "sleep_quality", "mental_health_score"],
        "target_column": ["sleep_quality", "mental_health_score"],
        "model_type": "random_forest",  # or 'xgboost', 'logistic', etc.
        "test_size": 0.2
    }

    mental_health_training = MentalHealthProcessor(config["csv_file_path"], "keras")

    # Choose your framework based on these guidelines:

    # Use scikit-learn when:
    # - Small to medium dataset (<100K samples)
    # - Tabular/structured data
    # - Need fast prototyping
    # - Want interpretable models
    # - Limited computational resources

    print("Training with scikit-learn (Recommended for tabular data):")
    sklearn_model, X_test_sk, y_test_sk, pred_sk, _ = mental_health_training.complete_workflow(
        config, framework='sklearn'
    )
    # Save everything
    joblib.dump(sklearn_model, 'models/mental_health_pipeline.pkl')

    print("\\n" + "="*60)
    print("Training with Keras (Recommended for complex patterns):")
    keras_model, X_test_ke, y_test_ke, pred_ke, history = mental_health_training.complete_workflow(
        config, framework='keras'
    )