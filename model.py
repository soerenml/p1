import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3


def model_1():
    """"
        Returns a CNN.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )
    return model


def model_2():
    local_weights_file = '/Users/soeren/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # Use the inceptionV3 model architecture
    # Important: the model is not trained yet! We just got the architecture.

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  # define input shape
                                    include_top=False,  # exclude top layer
                                    weights=None)  # deactivate default weights

    # Load weights from already trained model into our InceptionV3
    pre_trained_model.load_weights(local_weights_file)

    # Lock layers to prevent re-training
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Summarize the model
    pre_trained_model.summary()

    # We take the conv. layers mixed7 with (7,7,768)
    last_layer = pre_trained_model.get_layer('mixed7').output

    # Build the final model
    # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_layer)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(pre_trained_model.input, x)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
