# General training algorithm using tensorFlow
import pickle
import numpy
import random
import matplotlib.pyplot as pylot
import cv2

def compile_training_data(data_dir_path: str, categories: list[str], img_size: int, convert_to_grayscale: bool = True, debug: bool = False):
    training_data = create_training_data(data_dir_path, categories, img_size, convert_to_grayscale, debug)
    color_values = 3
    if convert_to_grayscale:
        color_values = 1
    
    return prepare_training_data(training_data, img_size, color_values)


# Categories should be subfolders
def create_training_data(data_dir_path: str, categories: list[str], img_size: int, convert_to_grayscale: bool = True, debug: bool = False):
    # img_colors = cv2.IMREAD_COLOR
    # if convert_to_grayscale:
    #     img_colors = cv2.IMREAD_GRAYSCALE

    training_data = []

    for category in categories:
        path = os.path.join(data_dir_path, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                if convert_to_grayscale:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                
                resized_array = cv2.resize(img_array, (img_size, img_size))
                
                if convert_to_grayscale:
                    training_data.append([resized_array, class_num])#[class_num, class_num, class_num]])
                else:
                    training_data.append([resized_array, [class_num, class_num, class_num]])

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

    for path in img_paths:
        if not os.path.isfile(path):
            print(f"WARNING: {path} doesn't lead to a file!")
            continue
        
        try:
            img_array = cv2.imread(path)
            if convert_to_grayscale:
                img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

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

