import tensorflow as tf
import tensorflow.keras.layers as layers

def create_policy_net(obs_shape, num_actions, dense_units=128, depth=2, 
                      activation="tanh"):
    inputs = layers.Input(shape=obs_shape)
    x = inputs
    current_units = dense_units
    for _ in range(depth):
        x = layers.Dense(current_units, activation=activation)(x)
        current_units = max(current_units // 2, 32)
    outputs = layers.Dense(num_actions, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_value_net(state_shape, dense_units=128, depth=2, 
                    activation="tanh"):
    inputs = layers.Input(shape=state_shape)
    x = inputs
    current_units = dense_units
    for _ in range(depth):
        x = layers.Dense(current_units, activation=activation)(x)
        current_units = max(current_units // 2, 32)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)