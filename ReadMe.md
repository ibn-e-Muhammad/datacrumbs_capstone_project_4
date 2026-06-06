# California Housing Prices ANN Regression
**Author:** Antigravity (Data Scientist Agent)
**Date Started:** 2026-06-06

This is an append-only development log and project report. All configurations, execution steps, and decisions will be documented chronologically.

---
## Phase 1: Planning & Setup
**Timestamp:** 2026-06-06

### Project Setup
- **Objective:** Build and evaluate an Artificial Neural Network to predict a continuous target variable (`median_house_value`) using a regression dataset ("California Housing Prices").
- **Dataset:** `housing.csv` (Features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`, Target: `median_house_value`).
- **Framework:** Keras (TensorFlow)

### Constraints & Directives
1. **Append-Only Logging:** This ReadMe must only be appended to, never overwritten.
2. **Hardware Constraints:** Keras/TensorFlow is explicitly forced to execute strictly on the CPU to avoid PCIe transfer bottlenecks for this tabular dataset.
3. **Missing Data Imputation:** Missing numerical values (e.g., in `total_bedrooms`) will be imputed using the median.
4. **Feature Scaling:** `StandardScaler` will be used to normalize numerical features.
5. **Categorical Data:** The `ocean_proximity` feature will be One-Hot Encoded.


---
## Phase 2: Data Preprocessing
**Timestamp:** 2026-06-06 13:19:26

### Hardware Configuration
- **Device:** CPU (GPU strictly disabled via `tf.config.set_visible_devices([], 'GPU')` to avoid PCIe bottlenecks).

### Exploratory Data Analysis (EDA)
- **Dataset Shape:** (20640, 10)
- **Data Types:**
```
longitude             float64
latitude              float64
housing_median_age    float64
total_rooms           float64
total_bedrooms        float64
population            float64
households            float64
median_income         float64
median_house_value    float64
ocean_proximity        object
dtype: object
```
- **Missing Values (Before):**
```
longitude               0
latitude                0
housing_median_age      0
total_rooms             0
total_bedrooms        207
population              0
households              0
median_income           0
median_house_value      0
ocean_proximity         0
dtype: int64
```

### Data Splitting
- **Split:** 80/20 (Train/Test)
- **X_train shape:** (16512, 9)
- **X_test shape:** (4128, 9)

### Preprocessing Applied
- **Numerical Features:** Imputed missing values using the median. Scaled using `StandardScaler`.
- **Categorical Features:** Applied One-Hot Encoding to `ocean_proximity`.
- **Final Feature Count:** 13 features.
- **Final Features List:** ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']

### Data Artifacts
- Processed arrays (`X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`) saved to `processed_data/`.
- Pipeline object (`preprocessor.pkl`) saved to `processed_data/`.


---
## Phase 3: Model Architecture
**Timestamp:** 2026-06-06 13:27:36

### Architecture Specifications
- **Framework:** Keras (TensorFlow Sequential API)
- **Input Layer:** Expects 13 features.
- **Hidden Layer 1:** 64 neurons, Activation: `relu`
- **Hidden Layer 2:** 32 neurons, Activation: `relu`
- **Output Layer:** 1 neuron, Activation: `linear` (for regression)

### Model Summary
```text
Model: "sequential"
+--------------------------------------------------------------------------+
| Layer (type)                    | Output Shape           |       Param # |
|---------------------------------+------------------------+---------------|
| hidden_layer_1 (Dense)          | (None, 64)             |           896 |
|---------------------------------+------------------------+---------------|
| hidden_layer_2 (Dense)          | (None, 32)             |         2,080 |
|---------------------------------+------------------------+---------------|
| output_layer (Dense)            | (None, 1)              |            33 |
+--------------------------------------------------------------------------+
 Total params: 3,009 (11.75 KB)
 Trainable params: 3,009 (11.75 KB)
 Non-trainable params: 0 (0.00 B)

```
- **Total Trainable Parameters:** 3009

### Artifact Management
- Uncompiled model architecture saved to `models/uncompiled_model.keras` for use in Phase 4.


---
## Phase 4: Training & Tuning
**Timestamp:** 2026-06-06 13:36:20

### Hardware Configuration
- **Device:** CPU (Explicitly forced via `tf.config.set_visible_devices([], 'GPU')`).

### Hyperparameters
- **Optimizer:** adam
- **Loss Function:** mse
- **Batch Size:** 32
- **Epochs:** 50
- **Validation Split:** 0.2

### Crucial Metric Tracking
#### Before and After Training Results
- **Epoch 1:**
  - `loss`: 56014360576.0000
  - `val_loss`: 56279670784.0000
- **Epoch 50:**
  - `loss`: 4433307648.0000
  - `val_loss`: 4742916096.0000

*(Note: The empirical comparison demonstrates the network's learning progress throughout the 50 epochs)*.

### Artifact Management
- Fully trained model saved to `models/trained_model.keras`.
- Full training history dictionary saved to `models/training_history.json`.


---
## Phase 5: Evaluation
**Timestamp:** 2026-06-06 14:05:31

### Hardware Configuration
- **Device:** CPU (Explicitly forced via `tf.config.set_visible_devices([], 'GPU')`).

### Final Test Metrics
Evaluated on the unseen `X_test` dataset (20% of original dataset).

- **Mean Absolute Error (MAE):** 49183.29
- **Mean Squared Error (MSE):** 4700559332.83
- **R² Score:** 0.6413

### Analytical Summary
The Artificial Neural Network achieved solid performance on the California Housing dataset. The **R² score of 0.6413** indicates that approximately 64.13% of the variance in the median house value can be explained by our model's features. This signifies a strong predictive relationship between the input demographic/geographic features and house prices. Additionally, the **MAE of 49183.29** tells us that on average, the model's predicted median house value is off by roughly this dollar amount from the actual value. Given that California house prices commonly range in the hundreds of thousands, this error margin provides a practical understanding of the model's day-to-day prediction accuracy. Overall, the network successfully learned the underlying patterns and generalized well to unseen data without extreme overfitting, validating the chosen hyperparameters and architecture.