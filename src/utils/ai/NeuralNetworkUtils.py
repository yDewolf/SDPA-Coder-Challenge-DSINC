# General training algorithm using tensorFlow
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy
import random
import matplotlib.pyplot as pylot
import cv2


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

# Strides is basically the grid of pixels it will look for etc etc
# Input shape -> X.shape[1:]
def create_model_convolutional(output_layer_size: int, kernel_sizes: list[int], strides: list[tuple], pool_sizes: list[tuple], input_shape):
    if len(kernel_sizes) != len(strides):
        # Push error:
        print("WARNING: Kernel sizes list should be the same size as strides list")
        return

    new_model = tensorflow.keras.models.Sequential()
    
    # Hidden layers and input layer
    for idx in range(len(kernel_sizes)):
        if idx == 0:
            new_model.add(Conv2D(kernel_sizes[idx], strides[idx], input_shape=input_shape))
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


def model_predict(model, categories: list[str], image_paths: list[str] = []):
    predict_data = create_predict_data(image_paths, convert_to_grayscale=False)
    X = prepare_predict_data(predict_data, 32)
    #X = X/255.0

    predictions = model.predict(X)

    for idx in range(len(image_paths)):
        highest_guess = predictions[idx] * len(categories)
        img_array = cv2.imread(image_paths[idx])

        pylot.imshow(img_array)
        pylot.title(f"Guess: {categories[int(highest_guess)]}")
        print(categories[int(highest_guess)], highest_guess[0])
        pylot.show()


def compile_training_data(data_dir_path: str, categories: list[str], img_size: int, convert_to_grayscale: bool = True, debug: bool = False):
    training_data = create_training_data(data_dir_path, categories, img_size, convert_to_grayscale, debug)
    color_values = 3
    if convert_to_grayscale:
        color_values = 1
    return prepare_training_data(training_data, img_size, color_values)


# Categories should be subfolders
def create_training_data(data_dir_path: str, categories: list[str], img_size: int, convert_to_grayscale: bool = True, debug: bool = False):
    img_colors = cv2.IMREAD_ANYCOLOR
    if convert_to_grayscale:
        img_colors = cv2.IMREAD_GRAYSCALE

    training_data = []

    for category in categories:
        path = os.path.join(data_dir_path, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), img_colors)
                resized_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([resized_array, class_num])

                if debug:
                    pylot.imshow(img_array)
                    pylot.show()
                    if int(input()) > 0:
                        debug = False
            
            except Exception as e:
                pass
    
    random.shuffle(training_data)
    return training_data


def create_predict_data(img_paths: list[str], img_size: int = 32, convert_to_grayscale: bool = True, debug: bool = False):
    predict_data = []
    img_colors = cv2.IMREAD_ANYCOLOR
    if convert_to_grayscale:
        img_colors = cv2.IMREAD_GRAYSCALE

    for path in img_paths:
        if not os.path.isfile(path):
            print(f"WARNING: {path} doesn't lead to a file!")
            continue
        
        try:
            img_array = cv2.imread(path, img_colors)
            resized_array = cv2.resize(img_array, (img_size, img_size))
            # Should this append a list?
            predict_data.append(resized_array)
           
            if debug:
                pylot.imshow(img_array)
                pylot.show()
                if int(input()) > 0:
                    debug = False
        
        except Exception as e:
            pass
    
    return predict_data

def prepare_predict_data(predict_data, img_size: int, color_values: int = 1):
    X = []
    for features in predict_data:
        X.append(features)

    X = numpy.array(predict_data).reshape(-1, img_size, img_size, color_values)

    return X


def prepare_training_data(training_data, img_size: int, color_values: int = 1):
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    # The 3 in the end is because it's using rgb and not grayscale (1)
    X = numpy.array(X).reshape(-1, img_size, img_size, color_values)
    y = numpy.array(y)

    return X, y


def pickle_save_training_data(X_filename: str, y_filename: str, X, y):
    pickle_out = open(X_filename + ".pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(y_filename + ".pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def pickle_load_training_data(X_file_path: str, y_file_path: str):
    pickle_in = open(X_file_path, "rb")
    X = pickle.load(pickle_in)
    X = X/255.0
    pickle_in.close()

    pickle_in = open(y_file_path, "rb")
    y = pickle.load(pickle_in)
    pickle_in.close()

    return X, y


# def test_compile():
#     X, y = compile_training_data("training_data", ["alien_ducks", "normal_ducks"], 32)
#     pickle_save_training_data("X1", "y1", X, y)

# def test_model():
#     model = tensorflow.keras.models.load_model("DuckDetection.keras")
#     image_paths = []
#     dir_path = "training_data/validate"
#     count = 0
#     max_images = 10
#     for img_path in os.listdir(dir_path):
#         if count >= max_images:
#             break

#         image_paths.append(dir_path + "/" + img_path)
#         count += 1

#     model_predict(model,["Alien Duck", "Normal Duck"], image_paths)



# test_model()