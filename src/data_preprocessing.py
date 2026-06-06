import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
import datetime

def main():
    readme_path = "../ReadMe.md"
    data_path = "../housing.csv"
    output_dir = "../processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(readme_path, "a") as f:
        f.write("\n\n---\n## Phase 2: Data Preprocessing\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Hardware Execution Constraint
        tf.config.set_visible_devices([], 'GPU')
        f.write("### Hardware Configuration\n")
        f.write("- **Device:** CPU (GPU strictly disabled via `tf.config.set_visible_devices([], 'GPU')` to avoid PCIe bottlenecks).\n\n")

        # 2. Load Dataset
        df = pd.read_csv(data_path)
        f.write("### Exploratory Data Analysis (EDA)\n")
        f.write(f"- **Dataset Shape:** {df.shape}\n")
        f.write("- **Data Types:**\n```\n" + str(df.dtypes) + "\n```\n")
        f.write("- **Missing Values (Before):**\n```\n" + str(df.isnull().sum()) + "\n```\n\n")

        # Define columns
        target = "median_house_value"
        categorical_features = ["ocean_proximity"]
        numerical_features = [col for col in df.columns if col not in categorical_features + [target]]

        # Separate X and y
        X = df.drop(columns=[target])
        y = df[target]

        # 3. Preprocessing Pipelines
        # Numerical: Impute median -> Scale
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical: One-Hot Encode
        cat_pipeline = Pipeline([
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        # Combine
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ])

        # 4. Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        f.write("### Data Splitting\n")
        f.write(f"- **Split:** 80/20 (Train/Test)\n")
        f.write(f"- **X_train shape:** {X_train.shape}\n")
        f.write(f"- **X_test shape:** {X_test.shape}\n\n")

        # 5. Apply Preprocessing
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Extract feature names after encoding for reference
        # Note: Depending on scikit-learn version, get_feature_names_out might be used
        try:
            cat_encoder = preprocessor.named_transformers_['cat']['encoder']
            cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
            all_features = numerical_features + list(cat_features_encoded)
        except Exception as e:
            all_features = ["Feature_"+str(i) for i in range(X_train_processed.shape[1])]

        f.write("### Preprocessing Applied\n")
        f.write("- **Numerical Features:** Imputed missing values using the median. Scaled using `StandardScaler`.\n")
        f.write("- **Categorical Features:** Applied One-Hot Encoding to `ocean_proximity`.\n")
        f.write(f"- **Final Feature Count:** {X_train_processed.shape[1]} features.\n")
        f.write(f"- **Final Features List:** {all_features}\n\n")

        # 6. Save Processed Data
        np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)
        joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))

        f.write("### Data Artifacts\n")
        f.write("- Processed arrays (`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`) saved to `processed_data/`.\n")
        f.write("- Pipeline object (`preprocessor.pkl`) saved to `processed_data/`.\n")

    print("Data Preprocessing Complete. Check ReadMe.md for logs.")

if __name__ == "__main__":
    main()
