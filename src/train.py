import sys
from pathlib import Path
import pandas as pd

import tensorflow as tf

# Add data directory to path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
sys.path.append(str(project_root / 'data'))

data_path = project_root / 'data' / "midus_processed.csv"

def train():
    # load proccesed data
    df = pd.read_csv(data_path, sep=',')
    
    columns_to_drop = [
        'M2ID',
        'M2FAMNUM', 
        'SAMPLMAJ',
        'M2_LOG_C',
        'V7_A',
        'M2_LOG_I',
        'LOG_M2_N',
        'M2_DOPAM',
        'M2_EPINE',
        'M2_FIBRI',
        'M2_SICAM',
        'M3_EPISO',
        'M3_EXECU',
        'M2_EPISO',
        'M2_EXECU',
        'B1PAGE_M_scaled',
        'B1PGENDE_scaled',
        'episodic_change',
        'executive_change',
        'B1PAGE_M'
    ]
    
    df = df.drop(columns=columns_to_drop, axis=1)
    
    # split into input (X) and output (y) variables
    X_raw = df.drop(['is_stressed', 'is_inflammated'], axis=1) # 11 columns
    print(f'Columns({len(X_raw.columns)}) with reference:', X_raw.columns.tolist())
    X = X_raw.values
    y = df['is_stressed'].values  # binary target variable
    
    # define the keras model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(11, input_shape=(11,), activation='relu'),
        tf.keras.layers.Dense(8,  activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
       
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # train the keras model
    model.fit(X, y, epochs=150, batch_size=10)
        
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    
    # serialize model to JSON    
    model_json = model.to_json()
    with open(project_root / 'models' / "is_stressed_model.json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights(project_root / 'models' /"is_stressed_model.weights.h5")
    print("Saved model to disk")

    return True