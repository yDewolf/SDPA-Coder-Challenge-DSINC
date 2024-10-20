# General detection algorithm using tensorFlow and pre trained AI
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow
import cv2
from utils.DatasetUtils import create_predict_data, prepare_predict_data
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D # type: ignore


def create_model(output_layer_size: int, hidden_layer_sizes: list[int], hidden_layer_activations: list):
    if len(hidden_layer_activations) != len(hidden_layer_sizes):
        print("ERROR: Layer activations list and Layer sizes list should be the same length")
        return

    new_model = tensorflow.keras.models.Sequential()

    # Input layer
    new_model.add(Flatten())

    # Hidden layers
    for idx in range(len(hidden_layer_sizes)):
        new_model.add(tensorflow.keras.layers.Dense(hidden_layer_sizes[idx], activation=hidden_layer_activations[idx]))

    # Output layer
    new_model.add(tensorflow.keras.layers.Dense(output_layer_size, activation=tensorflow.nn.softmax))

    return new_model

# Strides are basically the grid of pixels it will look for etc etc
def create_model_convolutional(output_layer_size: int, kernel_sizes: list[int], strides: list[tuple], pool_sizes: list[tuple], input_shape):
    if len(kernel_sizes) != len(strides):
        # Push error:
        print("WARNING: Kernel sizes list should be the same size as strides list")
        return

    new_model = tensorflow.keras.models.Sequential()
    new_model.add(Conv2D(kernel_sizes[0], strides[0], input_shape=(32, 32, 3)))

    # Hidden layers and input layer
    for idx in range(len(kernel_sizes)):
        if idx == 0:
            continue
            # new_model.add(Conv2D(kernel_sizes[idx], strides[idx]))#, input_shape=input_shape))
        else:
            new_model.add(Conv2D(kernel_sizes[idx], strides[idx]))
        
        new_model.add(Activation("relu"))
        new_model.add(MaxPooling2D(pool_size=pool_sizes[idx]))
    
    # Pre output layer
    new_model.add(Flatten())
    new_model.add(Dense(kernel_sizes[-1]))

    # Output layer
    # 1 should be output_layer_size
    new_model.add(Dense(output_layer_size))
    new_model.add(Activation('sigmoid'))

    return new_model

def train_model_convolutional(model, X, y, epochs: int, loss: str = 'binary_crossentropy', optimizer:  str = 'adam', metrics: list[str] = ['accuracy']):
    train_model(model, X, y, epochs, optimizer, loss, metrics)

def train_model(model, x_train_data, y_train_data, _epochs: int, _optimizer: str = 'adam', _loss: str = 'sparse_categorical_crossentropy', _metrics: list[str] = ['accuracy']):
    # Training
    model.compile(
                optimizer=_optimizer,
                loss=_loss,
                metrics=_metrics
                )

    # Use binary with normal ducks X alien ducks
    model.fit(x_train_data, y_train_data, epochs=_epochs)


def model_predict(model, categories: list[str], image_paths: list[str] = [], convert_to_grayscale: bool = True):
    predict_data = create_predict_data(image_paths, convert_to_grayscale=convert_to_grayscale)
    color_values = 3
    if convert_to_grayscale:
        color_values = 1
    X = prepare_predict_data(predict_data, 32, color_values)
    #X = X/255.0

    predictions = model.predict(X)

    guesses = []
    for idx in range(len(image_paths)):
        highest_guess = predictions[idx][0]
        img_array = cv2.imread(image_paths[idx])

        guess_dict = {
            "highest_guess": highest_guess,
            "img_array": img_array
        }
        guesses.append(guess_dict)

        # pylot.imshow(img_array)
        # pylot.title(f"Guess: {categories[int(highest_guess)]}")
        print(categories[int(highest_guess)], highest_guess)
        # pylot.show()
    
    return guesses

