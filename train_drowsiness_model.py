# train_drowsiness_model.py
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from gtts import gTTS
import time
from speech_recognition_module import recognize_speech
import argparse

def generator(dir, gen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True),
              shuffle=True, batch_size=32, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                   class_mode=class_mode, target_size=target_size)

def build_model(input_shape, num_classes, conv1_filters, conv2_filters, dense_units, dropout_rate_conv, dropout_rate_dense):
    model = Sequential([
        Conv2D(conv1_filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(conv2_filters, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        BatchNormalization(),
        Dropout(dropout_rate_conv),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate_dense),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, valid_dir, batch_size, target_size, epochs, conv1_filters, conv2_filters,
                dense_units, dropout_rate_conv, dropout_rate_dense, feedback=True):
    train_batch = generator(train_dir, shuffle=True, batch_size=batch_size, target_size=target_size)
    valid_batch = generator(valid_dir, shuffle=True, batch_size=batch_size, target_size=target_size)
    SPE = len(train_batch.classes) // batch_size
    VS = len(valid_batch.classes) // batch_size
    print(SPE, VS)

    input_shape = (*target_size, 1)
    num_classes = len(train_batch.class_indices)

    model = build_model(input_shape, num_classes, conv1_filters, conv2_filters, dense_units, dropout_rate_conv, dropout_rate_dense)

    # Implementing callbacks
    checkpoint = ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    # Train the model
    model.fit_generator(train_batch, validation_data=valid_batch, epochs=epochs, steps_per_epoch=SPE,
                        validation_steps=VS, callbacks=[checkpoint, early_stopping, reduce_lr])

    # Save the final model
    model.save('models/final_model.h5', overwrite=True)

    if feedback:
        tts = gTTS("Training completed successfully.")
        tts.save("feedback.mp3")
        os.system("start feedback.mp3")

def interact_with_user():
    while True:
        command = recognize_speech()

        if command == 'train':
            train_model(args.train_dir, args.valid_dir, args.batch_size, args.target_size, args.epochs,
                        args.conv1_filters, args.conv2_filters, args.dense_units, args.dropout_rate_conv, args.dropout_rate_dense)
        elif command == 'pause':
            print("Training paused. Say 'resume' to continue.")
            tts = gTTS("Training paused. Say 'resume' to continue.")
            tts.save("feedback.mp3")
            os.system("start feedback.mp3")
            while True:
                resume_command = recognize_speech()
                if resume_command == 'resume':
                    break
        elif command == 'stop':
            print("Training stopped.")
            tts = gTTS("Training stopped.")
            tts.save("feedback.mp3")
            os.system("start feedback.mp3")
            break
        else:
            print("Invalid command. Please say 'train', 'pause', or 'stop'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Drowsiness Detection Model')
    parser.add_argument('--train_dir', required=True, help='Path to the training data directory')
    parser.add_argument('--valid_dir', required=True, help='Path to the validation data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--target_size', type=int, nargs=2, default=[24, 24], help='Target size for input images')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--conv1_filters', type=int, default=32, help='Number of filters in the first convolutional layer')
    parser.add_argument('--conv2_filters', type=int, default=64, help='Number of filters in the second convolutional layer')
    parser.add_argument('--dense_units', type=int, default=256, help='Number of units in the dense layer')
    parser.add_argument('--dropout_rate_conv', type=float, default=0.25, help='Dropout rate in convolutional layers')
    parser.add_argument('--dropout_rate_dense', type=float, default=0.5, help='Dropout rate in dense layers')

    args = parser.parse_args()

    interact_with_user()
