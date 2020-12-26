import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os
print(tf.__version__)

def model(args):
    # Define the source directories
    base_dir = args.DATA_PATH
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Rescale images
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0/255.
    )

    # training set
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )

    # validation set
    validation_gen = test_datagen.flow_from_directory(
        validation_dir,
        batch_size = args.BATCH_SIZE,
        class_mode = 'binary',
        target_size = (150, 150)
    )

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

    print(model.summary())

    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="/Users/soeren/tf/logs"
    )

    # Early stopping callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > 0.73):
                print("\n Stopped training: " + str(logs.get('accuracy')))
                self.model.stop_training = True

    accuracy_callback = myCallback()

    history = model.fit(
        train_gen,
        validation_data=validation_gen,
        steps_per_epoch=args.STEPS_EPOCHS,
        epochs=args.EPOCHS,
        validation_steps=10,
        callbacks=[tensorboard_callback, accuracy_callback]
    )
