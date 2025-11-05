import sys
from pathlib import Path
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add data directory to path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
sys.path.append(str(project_root / 'data'))

data_path = project_root / 'data' / "mh_processed.csv"

def train():
    # load proccesed data
    df = pd.read_csv(data_path, sep=',')

    
    # split into input (X) and output (y) variables
    # X_raw = df.drop(['mood_rating','stress_level', 'mental_health_score', 'weekly_anxiety_score', 'weekly_depression_score'], axis=1)
    X_raw = df.drop(['mental_health_score' ], axis=1)
    print(f'Columns({len(X_raw.columns)}) with reference:', X_raw.columns.tolist()) # 19 cols
    X = X_raw.values
    
    y_raw = df['mental_health_score']
    # print(f'Columns({len(y_raw.columns)}) with reference:', y_raw.columns.tolist()) # 5 cols
    y = y_raw.values
    
    # split the data into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # define the keras model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(23, input_shape=(23,), activation='relu'),
        tf.keras.layers.Dense(30,  activation='relu'),
        tf.keras.layers.Dense(30,  activation='relu'),
        tf.keras.layers.Dense(20,  activation='relu'),
        tf.keras.layers.Dense(16,  activation='relu'),
        tf.keras.layers.Dense(8,  activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
       
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the keras model
    model.fit(X_train, y_train, epochs=1000, batch_size=10)

    # evaluate the keras model
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    
    # # serialize model to JSON    
    # model_json = model.to_json()
    # with open(project_root / 'models' / "is_stressed_model.json", "w") as json_file:
    #     json_file.write(model_json)
        
    # # serialize weights to HDF5
    # model.save_weights(project_root / 'models' /"is_stressed_model.weights.h5")
    # print("Saved model to disk")

    return True

if __name__ == "__main__":
    train()