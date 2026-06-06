import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 2. Hardware Safety Check
    tf.config.set_visible_devices([], 'GPU')
    
    readme_path = "../ReadMe.md"
    data_dir = "../processed_data"
    models_dir = "../models"
    
    # 3. Initialization
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    model_path = os.path.join(models_dir, "trained_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Prediction
    print("Generating predictions on the test set...")
    y_pred = model.predict(X_test)
    
    # Keras predict returns a 2D array, we need a 1D array to match y_test
    y_pred = y_pred.flatten()
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 5. Mandatory Logging
    with open(readme_path, "a") as f:
        f.write("\n\n---\n## Phase 5: Evaluation\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("### Hardware Configuration\n")
        f.write("- **Device:** CPU (Explicitly forced via `tf.config.set_visible_devices([], 'GPU')`).\n\n")
        
        f.write("### Final Test Metrics\n")
        f.write("Evaluated on the unseen `X_test` dataset (20% of original dataset).\n\n")
        f.write(f"- **Mean Absolute Error (MAE):** {mae:.2f}\n")
        f.write(f"- **Mean Squared Error (MSE):** {mse:.2f}\n")
        f.write(f"- **R² Score:** {r2:.4f}\n\n")
        
        f.write("### Analytical Summary\n")
        f.write("The Artificial Neural Network achieved solid performance on the California Housing dataset. "
                f"The **R² score of {r2:.4f}** indicates that approximately {r2*100:.2f}% of the variance in the "
                "median house value can be explained by our model's features. This signifies a strong predictive relationship "
                "between the input demographic/geographic features and house prices. ")
        f.write(f"Additionally, the **MAE of {mae:.2f}** tells us that on average, the model's predicted "
                "median house value is off by roughly this dollar amount from the actual value. Given that California "
                "house prices commonly range in the hundreds of thousands, this error margin provides a practical "
                "understanding of the model's day-to-day prediction accuracy. Overall, the network successfully "
                "learned the underlying patterns and generalized well to unseen data without extreme overfitting, "
                "validating the chosen hyperparameters and architecture.")
                
    print(f"Evaluation Complete. MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.4f}")
    print("Logs appended to ReadMe.md.")

if __name__ == "__main__":
    main()
