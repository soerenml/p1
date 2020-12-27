import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import os
import time

print(tf.__version__)


def model(args):
    # Define the source directories
    base_dir = args.DATA_PATH
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Rescale images
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.
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
        batch_size=args.BATCH_SIZE,
        class_mode='binary',
        target_size=(150, 150)
    )

    # Import model type
    import model

    if args.MODEL == "model_1":
        print("CNN")
        model = model.model_1()
        print(model.summary())
    if args.MODEL == "model_2":
        print("Inception V3")
        model = model.model_2()
        print(model.summary())

    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Save logs for Tensorboard
    tf_log_dir="/Users/soeren/tf-logs/{}".format(args.RUN_ID)
    print("Logs saved under: {}".format(tf_log_dir))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tf_log_dir
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
        validation_steps=11,
        callbacks=[tensorboard_callback, accuracy_callback]
    )
