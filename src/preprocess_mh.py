import sys
from pathlib import Path
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add data directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'data'))

data_path = project_root / 'data' / "digital_diet_mental_health.csv"

if __name__ == "__main__":
    df = pd.read_csv(data_path, sep=',')
    
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['location_type'] = df['location_type'].map({'Suburban': 0, 'Rural': 1, 'Urban': 2})
    
    df.drop(columns=['user_id'], axis=1, inplace=True)
    
    df.to_csv(project_root / 'data' / 'mh_processed.csv', index=False)
    
    