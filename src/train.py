import os
import json
import datetime
import numpy as np
import tensorflow as tf

def main():
    # 2. Hardware Safety Check
    tf.config.set_visible_devices([], 'GPU')
    
    readme_path = "../ReadMe.md"
    data_dir = "../processed_data"
    models_dir = "../models"
    
    # 3. Initialization
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    
    model_path = os.path.join(models_dir, "uncompiled_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Compilation & Training Parameters
    optimizer = "adam"
    loss_function = "mse"
    batch_size = 32
    epochs = 50
    validation_split = 0.2
    
    model.compile(optimizer=optimizer, loss=loss_function)
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )
    
    # 5. Artifact Management
    trained_model_path = os.path.join(models_dir, "trained_model.keras")
    model.save(trained_model_path)
    
    # Convert history.history to standard float types for JSON serialization
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    history_path = os.path.join(models_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)
        
    # Extract Epoch 1 and Epoch 50 metrics
    # Note: history.history stores lists starting from index 0 (Epoch 1) to index 49 (Epoch 50)
    epoch_1_loss = history_dict['loss'][0]
    epoch_1_val_loss = history_dict['val_loss'][0]
    
    epoch_50_loss = history_dict['loss'][-1]
    epoch_50_val_loss = history_dict['val_loss'][-1]
    
    # 6. Mandatory Logging
    with open(readme_path, "a") as f:
        f.write("\n\n---\n## Phase 4: Training & Tuning\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("### Hardware Configuration\n")
        f.write("- **Device:** CPU (Explicitly forced via `tf.config.set_visible_devices([], 'GPU')`).\n\n")
        
        f.write("### Hyperparameters\n")
        f.write(f"- **Optimizer:** {optimizer}\n")
        f.write(f"- **Loss Function:** {loss_function}\n")
        f.write(f"- **Batch Size:** {batch_size}\n")
        f.write(f"- **Epochs:** {epochs}\n")
        f.write(f"- **Validation Split:** {validation_split}\n\n")
        
        f.write("### Crucial Metric Tracking\n")
        f.write("#### Before and After Training Results\n")
        f.write(f"- **Epoch 1:**\n")
        f.write(f"  - `loss`: {epoch_1_loss:.4f}\n")
        f.write(f"  - `val_loss`: {epoch_1_val_loss:.4f}\n")
        f.write(f"- **Epoch 50:**\n")
        f.write(f"  - `loss`: {epoch_50_loss:.4f}\n")
        f.write(f"  - `val_loss`: {epoch_50_val_loss:.4f}\n\n")
        f.write("*(Note: The empirical comparison demonstrates the network's learning progress throughout the 50 epochs)*.\n\n")
        
        f.write("### Artifact Management\n")
        f.write("- Fully trained model saved to `models/trained_model.keras`.\n")
        f.write("- Full training history dictionary saved to `models/training_history.json`.\n")

    print("Training Complete. Check ReadMe.md for logs.")

if __name__ == "__main__":
    main()
