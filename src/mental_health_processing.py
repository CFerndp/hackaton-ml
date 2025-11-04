import sys
from pathlib import Path

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Add data directory to path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
sys.path.append(str(project_root / 'data'))

data_path = project_root / "data" / "digital_diet_mental_health.csv"

class MentalHealthProcessor:

    def __init__(self, config: dict, framework: str):
        self.config = config
        self.framework = framework
        self.df = pd.read_csv(data_path)
        # Create results directory if it doesn't exist
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)


    def process_dataset(
            self,
            column_to_drop: str | list,
            columns_to_restructure,
            columns_to_normalize,
            mapping_dict,
            columns_to_move,
            target_column
    ):
        """"
        Process a CSV dataset with multiple operations:
        1. Drop specified columns
        2. Restructure string columns to numerical values
        3. Move specified columns to the end
        4. Split into input (X) and output (y) variables
        5. Handle non-continuous columns

        Parameters:
        - csv_file_path: Path to the CSV file
        - column_to_drop: Column name to drop (string or list)
        - columns_to_restructure: Columns name to convert from string to numerical
        - mapping_dict: Dictionary for value mapping (e.g., {"a": 1, "b": 2, "c": 3})
        - columns_to_move: List of column names to move to the end
        - target_column: Column name for the output variable (y)
        Returns:
        - X: Input features (DataFrame)
        - y: Output variable (Series)
        - processed_df: Fully processed DataFrame
        """

        # 1. Load the dataset
        df = pd.read_csv(data_path)
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")

        # 2. Drop specified column(s)
        if isinstance(column_to_drop, str):
            column_to_drop = [column_to_drop]

        columns_exist = [col for col in column_to_drop if col in df.columns]
        if columns_exist:
            df = df.drop(columns=columns_exist)
            print(f"Dropped columns: {columns_exist}")
        else:
            print("No specified columns found to drop")

        # 3. Restructure string column to numerical values
        for column_to_restructure in columns_to_restructure:
            if column_to_restructure in df.columns:
                df[column_to_restructure] = df[column_to_restructure].map(mapping_dict[column_to_restructure])
                # Handle any values not in mapping dictionary
                if df[column_to_restructure].isna().any():
                    print(f"Warning: Some values in '{column_to_restructure}' not found in mapping dictionary")
                    # Option 1: Fill with a default value (e.g., -1)
                    df[column_to_restructure] = df[column_to_restructure].fillna(-1)
                    # Option 2: Alternatively, you can raise an error or use different strategy
                print(f"Restructured column '{column_to_restructure}' using mapping: {mapping_dict}")
            else:
                print(f"Column '{column_to_restructure}' not found for restructuring")

        # 4. Move specified columns to the end
        columns_to_move = [col for col in columns_to_move if col in df.columns]
        if columns_to_move:
            other_columns = [col for col in df.columns if col not in columns_to_move]
            new_column_order = other_columns + columns_to_move
            df = df[new_column_order]
            print(f"Moved columns to the end: {columns_to_move}")
        else:
            print("No specified columns found to move")

        # x. Manage missing values
        # Check for missing values
        print(f"total null value in columns:\n {df.isnull().sum()}")
        print(f"percentage of null values in columns:\n {df.isnull().sum() / len(df) * 100}")  # percentage

        # Visualize missing values
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
        plt.savefig(self.results_dir / 'mental_health_missing_data_map.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Visualize correlation
        plt.figure(figsize=(20, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation")
        plt.savefig(self.results_dir / 'mental_health_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

        # y. Normalize columns
        # Apply normalization to multiple columns
        # Min-Max normalization
        df[columns_to_normalize] = df[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # 5. Identify non-continuous (categorical) columns for potential encoding
        non_continuous_columns = self.identify_non_continuous_columns(df)
        print(f"Non-continuous columns identified: {non_continuous_columns}")

        # 6. Split into input (X) and output (y) variables
        if isinstance(target_column, list):
            assert all(t in df.columns for t in target_column)
        # if target_column in df.columns:
        inputs = df.drop(columns=target_column if isinstance(target_column, list) else [target_column])
        output = df[target_column]
        print(f"Split dataset - X shape: {inputs.shape}, y shape: {output.shape}")

        # WARNING: wait for this code for other column than mental_health_score
        # if non_continuous_columns:
        #     X_encoded = self.encode_categorical_variables(inputs, non_continuous_columns, encoding_strategy='onehot')
        #     print(f"X after encoding shape: {X_encoded.shape}")

        # Display results
        print("\\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Final X shape: {inputs.shape}")
        print(f"Final y shape: {output.shape}")
        print(f"Processed DataFrame shape: {df.shape}")
        print(f"INPUTS columns: {list(inputs.columns)}")
        print(f"OUTPUTS columns: {list(output.columns)}")
        # print(f"OUTPUT COLUMN NAME: {output.name}")

        # Show first few rows of processed data
        print("\\nFirst 5 rows of processed DataFrame:")
        print(df.head())

        return inputs, output, df

    def normalize_columns(self, columns_to_normalize):
        other_features = [col for col in self.df.columns if col not in columns_to_normalize]

        # Create transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), columns_to_normalize),
                ('other', 'passthrough', other_features)  # keep other columns unchanged
            ]
        )

        # Apply transformation
        df_normalized = preprocessor.fit_transform(self.df)
        # Convert back to DataFrame if needed
        df_normalized = pd.DataFrame(df_normalized, columns=columns_to_normalize + other_features)

        return df_normalized

    def identify_non_continuous_columns(self, df: DataFrame, threshold=10):
        """
        Identify non-continuous (categorical) columns in the dataset.

        Parameters:
        - df: DataFrame
        - threshold: Maximum number of unique values to consider as categorical

        Returns:
        - List of non-continuous column names
        """
        non_continuous_cols = []

        for column in df.columns:
            # Skip if all values are numeric and many unique values
            if pd.api.types.is_numeric_dtype(df[column]):
                unique_ratio = df[column].nunique() / len(df[column])
                if df[column].nunique() <= threshold and unique_ratio < 0.5:
                    non_continuous_cols.append(column)
            else:
                # Non-numeric columns are definitely non-continuous
                non_continuous_cols.append(column)

        return non_continuous_cols

    def encode_categorical_variables(self, X, categorical_columns, encoding_strategy='onehot'):
        """
        Encode categorical variables using specified strategy.

        Parameters:
        - X: Input features DataFrame
        - categorical_columns: List of categorical column names
        - encoding_strategy: 'onehot' for one-hot encoding or 'label' for label encoding

        Returns:
        - X_encoded: Encoded DataFrame
        - encoder: Fitted encoder object for future transformations
        """
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.compose import ColumnTransformer

        X_encoded = X.copy()

        if encoding_strategy == 'onehot':
            # One-hot encoding for categorical variables
            encoder = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
                ],
                remainder='passthrough'
            )
            X_encoded = pd.DataFrame(encoder.fit_transform(X_encoded))

        elif encoding_strategy == 'label':
            # Label encoding for each categorical column
            encoders = {}
            for col in categorical_columns:
                if col in X_encoded.columns:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    encoders[col] = le

            print(f"Applied {encoding_strategy} encoding to categorical variables")
        return X_encoded

    # Choose your approach based on your needs:
    # - scikit-learn: Traditional ML, faster training, better for tabular data
    def build_model_with_sklearn(self, inputs, output, model_type='random_forest', test_size=0.2, random_state=42):
        """
        Build and train models using scikit-learn (Recommended for tabular data)
        Parameters:
        - inputs: Features DataFrame,
        - output: Target variable,
        - model_type: 'random_forest', 'xgboost', 'logistic', 'svm',
        - test_size: Proportion for test set,

        Returns:,
        - trained_model: Fitted model
        - X_test, y_test: Test data
        - predictions: Model predictions
        """

        # 1. Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            inputs, output, test_size=test_size, random_state=random_state, stratify=output if len(np.unique(output)) < 10 else None
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # 2. Handle categorical variables in X
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns

        print(f"Categorical columns: {list(categorical_cols)}")
        print(f"Numerical columns: {list(numerical_cols)}")

        # Encode categorical variables\n",
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ])

        # 3. Choose and train model
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            # if output.dtype == 'object' or len(np.unique(output)) < 10:
            if len(np.unique(output)) < 10:
                model = RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=random_state,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=random_state
                )

        elif model_type == 'xgboost':
            from xgboost import XGBClassifier, XGBRegressor
            if output.dtype == 'object' or len(np.unique(output)) < 10:
                model = XGBClassifier(random_state=random_state, eval_metric='logloss')
            else:
                model = XGBRegressor(random_state=random_state)

        elif model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=random_state, max_iter=1000)

        elif model_type == 'svm':
            from sklearn.svm import SVC
            model = SVC(random_state=random_state, probability=True)

        # Create pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # 4. Train the model
        print(f"Training {model_type} model...")
        pipeline.fit(X_train, y_train)

        # 5. Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None

        # 6. Evaluate model
        self.evaluate_model(y_test, y_pred, y_pred_proba, model_type)

        return pipeline, X_test, y_test, y_pred

    def build_model_with_keras(
            self,
            inputs,
            output,
            model_type='dense',
            test_size=0.3,
            random_state=42,
            epochs=150,
            batch_size=10
    ):
        """
        Build and train models using Keras/TensorFlow (Recommended for complex patterns, large datasets)

        Parameters:
        - X: Features DataFrame
        - y: Target variable
        - model_type: 'dense', 'cnn', 'lstm' (for sequential data)
        - test_size: Proportion for test set
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # 1. Preprocess data for neural network
        X_processed = inputs.copy()

        # Encode categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns

        # One-hot encode categorical features
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)

        # Scale numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

        # Encode target variable
        # if output.dtype == 'object' or len(np.unique(output)) < 10:
        if len(np.unique(output)) < 10:
            # Classification problem
            le = LabelEncoder()
            y_encoded = le.fit_transform(output)
            num_classes = len(np.unique(y_encoded))
            problem_type = 'classification'
            # For binary classification, use sigmoid; for multi-class, use softmax
            if num_classes == 2:
                output_activation = 'sigmoid'
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                output_activation = 'softmax'
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            # Regression problem
            y_encoded = output.values
            num_classes = 1
            problem_type = 'regression'
            output_activation = 'linear'
            loss = 'mse'
            metrics = ['mae']

        # 2. Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=test_size, random_state=random_state
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Problem type: {problem_type}, Number of classes: {num_classes}")

        # 3. Build the neural network
        model = keras.Sequential()

        # Input layer
        model.add(layers.Dense(23, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(layers.Dropout(0.3))

        # Hidden layers
        model.add(layers.Dense(12, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(12, activation='relu'))
        model.add(layers.Dropout(0.2))

        # Output layer
        if problem_type == 'classification':
            if num_classes == 2:
                units = 1
                # model.add(layers.Dense(1, activation=output_activation))
            else:
                units = num_classes
                # model.add(layers.Dense(num_classes, activation=output_activation))
        else:
            units = 1
            # model.add(layers.Dense(1, activation=output_activation))
        print(f"units = {units}")
        model.add(layers.Dense(2, activation=output_activation))

        # 4. Compile the model
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )

        print("Model architecture:")
        model.summary()

        # 5. Train the model
        print("Training neural network...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        # 6. Evaluate the model
        test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
        print(f"\\nTest {metrics[0]}: {test_metric:.4f}")

        # 7. Make predictions
        y_pred = model.predict(X_test)

        if problem_type == 'classification':
            if num_classes == 2:
                y_pred_class = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_class = np.argmax(y_pred, axis=1)
            self.evaluate_model(y_test, y_pred_class, y_pred, 'neural_network')
        else:
            # For regression, calculate additional metrics
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"Regression Metrics - MSE: {mse:.4f}, R²: {r2:.4f}")

        # Plot training history
        self.plot_training_history(history, problem_type)

        return model, X_test, y_test, y_pred, history

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name=""):
        """Evaluate model performance"""
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        print(f"\\n{'=' * 50}")
        print(f"EVALUATION RESULTS for {model_name.upper()}")
        print(f"{'=' * 50}")

        # Check if classification or regression
        # if y_true.dtype == 'object' or len(np.unique(y_true)) < 10:
        if len(np.unique(y_true)) < 10:
            # Classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\\nClassification Report:")
            print(classification_report(y_true, y_pred))

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()

        else:
            # Regression metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")

    def plot_training_history(self, history, problem_type):
        """Plot training history for neural networks"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        if problem_type == 'classification':
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
        else:
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.ylabel('MAE')

        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'mental_health_keras_training.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Complete workflow example
    def complete_workflow(self, config: dict, framework='sklearn'):
        """
        Complete workflow from data loading to model training

        Parameters:
        - csv_file_path: Path to CSV file
        - config: Configuration dictionary
        - framework: 'sklearn' or 'keras'
        """


        inputs, output, processed_df = self.process_dataset(
            column_to_drop=config["column_to_drop"],
            columns_to_restructure=config["columns_to_restructure"],
            columns_to_normalize=config["columns_to_normalize"],
            mapping_dict=config["mapping_columns_to_map"],
            columns_to_move=config["columns_to_move"],
            target_column=config["target_column"],
        )

        print(f"\\nData processed successfully!")
        print(f"Features shape: {inputs.shape}")
        print(f"Target shape: {output.shape}")

        history = None

        # Build and train model
        if framework == 'sklearn':
            model, X_test, y_test, predictions = self.build_model_with_sklearn(
                inputs, output,
                model_type=config.get('model_type', 'random_forest'),
                test_size=config.get('test_size', 0.3)
            )
        else:
            model, X_test, y_test, predictions, history = self.build_model_with_keras(
                inputs, output,
                model_type=config.get('model_type', 'dense'),
                test_size=config.get('test_size', 0.3),
                epochs=config.get('epochs', 150)
            )

        return model, X_test, y_test, predictions, history