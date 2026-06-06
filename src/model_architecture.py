import tensorflow as tf
from tensorflow.keras import models, layers
import os
import datetime
import io

def build_model(input_dim):
    """Builds and returns the uncompiled ANN regression model."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='hidden_layer_1'),
        layers.Dense(32, activation='relu', name='hidden_layer_2'),
        layers.Dense(1, activation='linear', name='output_layer')
    ])
    return model

def main():
    readme_path = "../ReadMe.md"
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Feature dimension as specified in Phase 3
    input_dim = 13
    
    # 1. Hardware Execution Constraint (inherited from project settings)
    tf.config.set_visible_devices([], 'GPU')
    
    # 2. Build the model
    model = build_model(input_dim)
    
    # 3. Save the uncompiled model
    model_save_path = os.path.join(models_dir, "uncompiled_model.keras")
    model.save(model_save_path)
    
    # Extract model summary to a string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    
    # Count trainable parameters explicitly
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # 4. Append to ReadMe.md
    with open(readme_path, "a") as f:
        f.write("\n\n---\n## Phase 3: Model Architecture\n")
        f.write(f"**Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("### Architecture Specifications\n")
        f.write("- **Framework:** Keras (TensorFlow Sequential API)\n")
        f.write(f"- **Input Layer:** Expects {input_dim} features.\n")
        f.write("- **Hidden Layer 1:** 64 neurons, Activation: `relu`\n")
        f.write("- **Hidden Layer 2:** 32 neurons, Activation: `relu`\n")
        f.write("- **Output Layer:** 1 neuron, Activation: `linear` (for regression)\n\n")
        
        f.write("### Model Summary\n")
        f.write("```text\n")
        f.write(summary_string)
        f.write("```\n")
        f.write(f"- **Total Trainable Parameters:** {trainable_count}\n\n")
        
        f.write("### Artifact Management\n")
        f.write("- Uncompiled model architecture saved to `models/uncompiled_model.keras` for use in Phase 4.\n")

    print("Model Architecture Phase Complete. Check ReadMe.md for logs.")

if __name__ == "__main__":
    main()
