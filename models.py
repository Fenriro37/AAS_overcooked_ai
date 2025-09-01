import tensorflow as tf
import tensorflow.keras.layers as layers

def create_network(input_shape, model_type, num_actions=None, dense_units=128, depth=2, activation="tanh"):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    current_units = dense_units
    for _ in range(depth):
        x = layers.Dense(current_units, activation=activation)(x)
        current_units = max(current_units // 2, 32)

    if model_type == 'policy':
        if num_actions is None:
            raise ValueError("num_actions must be provided for model_type 'policy'")
        outputs = layers.Dense(num_actions, activation='softmax')(x)
    
    elif model_type == 'value':
        outputs = layers.Dense(1)(x)
        
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Must be 'policy' or 'value'.")

    return tf.keras.Model(inputs=inputs, outputs=outputs)