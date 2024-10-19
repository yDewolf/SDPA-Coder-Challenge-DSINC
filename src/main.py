import os
import utils.DatasetUtils as DatasetUtils
import utils.ModelUtils as ModelUtils

default_models_path: str = "models/"
default_pickle_data_path: str = "pickle_data/"
default_model_name: str = "DuckDetection"

X_train_path: str = default_pickle_data_path + "X_train.pickle"
y_train_path: str = default_pickle_data_path + "y_train.pickle"


# Neural Network functions:

def create_duck_model(X):
    shape = X.shape
    return ModelUtils.create_model_convolutional(3, [32, 16], [(3, 3), (3, 3)], [(2, 2), (2, 2)], shape)


def load_model(model_name: str):
    return ModelUtils.tensorflow.keras.models.load_model(default_models_path + model_name + ".keras")

def save_model(model, model_name: str):
    model.save(default_models_path + model_name + ".keras")


def compile_training_data(dir_path: str = "training_data/", categories: list[str] = ["alien_ducks", "normal_ducks"]):
    X, y = DatasetUtils.compile_training_data(dir_path, categories, 32, False)
    DatasetUtils.pickle_save_training_data(default_pickle_data_path + "X_train", default_pickle_data_path + "y_train", X, y)
    return X, y

def train_model(model, X_file_path: str, y_file_path: str, epochs: int = 3, loss = "binary_crossentropy", optimizer="adam"):
    X, y = DatasetUtils.pickle_load_training_data(X_file_path, y_file_path)

    DatasetUtils.train_model_convolutional(model, X, y, epochs, loss, optimizer)



# Menu functions

def load_model_menu(global_menu_variables: dict):
    model_name = input("Type the model name (without .keras): ")
    global_menu_variables["loaded_model"] = load_model(model_name)

def save_model_menu(global_menu_variables: dict):
    model_name = input("Type a name for the model: ")
    save_model(global_menu_variables["loaded_model"], model_name)

def train_model_menu(global_menu_variables: dict):
    epochs = int(input("How many iterations should the model be trained? (only round values greater than 0): "))

    train_model(global_menu_variables["loaded_model"], X_train_path, y_train_path, epochs)

def predict_images_menu(global_menu_variables: dict):
    categories = ["Alien Duck", "Normal Duck"]

    images_dir = ""
    while not os.path.isdir(images_dir):
        images_dir = input("Insert a images directory path: ")
        if not os.path.isdir(images_dir):
            # Push warning
            print("WARNING: Invalid directory path!")

    max_images = int(input("Type how many images should be predicted: "))

    image_paths = []

    # Get images from directory
    count = 0
    for img_path in os.listdir(images_dir):
        if count >= max_images:
            break

        image_paths.append(images_dir + "/" + img_path)
        count += 1

    guesses = DatasetUtils.model_predict(global_menu_variables["loaded_model"], categories, image_paths, False)
    for guess_dict in guesses:
        DatasetUtils.pylot.imshow(guess_dict["img_array"])
        DatasetUtils.pylot.title(f"Guess: {categories[int(guess_dict["highest_guess"])]}")
        DatasetUtils.pylot.show()

# Main menu

def test_main_menu(global_menu_variables):
    options = [
        {"name": "Exit"},
        {"name": "Load Model", "callable": load_model_menu},
        {"name": "Save Model", "callable": save_model_menu},
        {"name": "Train Model", "callable": train_model_menu},
        {"name": "Predict Images", "callable": predict_images_menu}
    ]
    for idx in range(len(options)):
        print(f"[{idx}] - {options[idx]["name"]}")

    option = -1
    while option < 0 or option > len(options):
        option = int(input("Select an option: "))
        
        if option < 0 or option > len(options):
            # Push warning
            print("WARNING: Invalid option!")
    
    if options[option].__contains__("callable"):
        options[option]["callable"](global_menu_variables)
        return True

    return False


def main(compile: bool = True):
    if not compile:
        X, y = compile_training_data()
        loaded_model = create_duck_model(X)
        save_model(loaded_model, "DuckDetection")
        return


    global_menu_variables = {
        "loaded_model": load_model(default_model_name)
    }

    # Menu loop
    running = True
    while running:
        running = test_main_menu(global_menu_variables)

main()