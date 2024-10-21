import os
import utils.DatasetUtils as DatasetUtils
import utils.ModelUtils as ModelUtils
from framework.MenuFramework import Menu, MenuHandler, MenuOption

default_models_path: str = "models/"
default_pickle_data_path: str = "pickle_data/"
default_model_name: str = "DuckDetection"

X_train_path: str = default_pickle_data_path + "X_train.pickle"
y_train_path: str = default_pickle_data_path + "y_train.pickle"

# Neural Network functions:

global loaded_model

def create_duck_model(X):
    return ModelUtils.create_model_convolutional(3, [64, 32], [(3, 3), (2, 2)], [(2, 2), (2, 2)], X.shape[1:])

def load_model(model_name: str, menu_handler: MenuHandler):
    global loaded_model
    loaded_model = ModelUtils.tensorflow.keras.models.load_model(default_models_path + model_name + ".keras")
    menu_handler.global_variables["loaded_model_name"] = model_name
    menu_handler.global_variables["is_model_loaded"] = True

def save_model(model, model_name: str):
    model.save(default_models_path + model_name + ".keras")

def compile_training_data(dir_path: str = "training_data/", categories: list[str] = ["alien_ducks", "normal_ducks"]):
    X, y = DatasetUtils.compile_training_data(dir_path, categories, 32, False)
    DatasetUtils.pickle_save_training_data(default_pickle_data_path + "X_train", default_pickle_data_path + "y_train", X, y)
    return X, y

def train_model(model, X_file_path: str, y_file_path: str, epochs: int = 3, loss = "binary_crossentropy", optimizer="adam"):
    X, y = DatasetUtils.pickle_load_training_data(X_file_path, y_file_path)

    ModelUtils.train_model_convolutional(model, X, y, epochs, loss, optimizer)



# Menu Conditions

def condition_loaded_model(menu_handler: MenuHandler):
    if menu_handler.global_variables.get("is_model_loaded"):
        return True
    
    return False

# Menu Updates

def update_load_model_menu(options: list[MenuOption], menu_handler: MenuHandler) -> list[MenuOption]:
    options = []
    for path in os.listdir(default_models_path):
        filename = path.replace(".keras", "")
        options.append(MenuOption(filename))

    return options

def update_main_menu_title(title: str, menu_handler: MenuHandler):
    if menu_handler.global_variables.__contains__("loaded_model_name"):
        title = f"Main menu | Loaded model: {menu_handler.global_variables["loaded_model_name"]}"
        return title
    
    return title

def update_predict_images_menu(options: list[MenuOption], menu_handler: MenuHandler) -> list[MenuOption]:
    options = []
    for path in os.listdir("training_data/"):
        #filename = path.replace(".keras", "")
        options.append(MenuOption("training_data/" + path))

    return options


# Menu Option Functions

def train_model_menu():
    global loaded_model
    epochs = int(input("Insert how many epochs the model should be trained: "))

    train_model(loaded_model, X_train_path, y_train_path, epochs)

def predict_single_image():
    image_path = ""
    # Certify that the path is actually a path
    while not os.path.isfile(image_path):
        image_path = input("Insert an image path: ")
        if not os.path.isfile(image_path):
            # Push Warning
            print(f"WARNING: {image_path} is not a valid image path!")


    categories: list[str] = ["Alien Duck", "Normal Duck"]
    guesses = ModelUtils.model_predict(loaded_model, categories, [image_path], False)
    for guess_dict in guesses:
        DatasetUtils.pylot.imshow(guess_dict["img_array"])
        DatasetUtils.pylot.title(f"Guess: {categories[guess_dict["highest_guess"]]}")
        DatasetUtils.pylot.show()

def save_model_menu():
    model_filename = input("Type the name of the model: ")

    global loaded_model
    save_model(loaded_model, model_filename)
    print("Saved model sucessfully!")

# Menu Custom Call Functions

def predict_images_in_dir(dir_path: str, menu_handler: MenuHandler):
    print(f"Predicting images from {dir_path}")
    max_images = int(input("Type how many images should be predicted: "))

    img_paths = []
    count: int = 0
    for img_path in os.listdir(dir_path):
        if count >= max_images:
            break
        
        img_paths.append(dir_path + "/" + img_path)
        count += 1
    
    #global loaded_model
    categories: list[str] = ["Alien Duck", "Normal Duck"]
    guesses = ModelUtils.model_predict(loaded_model, categories, img_paths, False)
    for guess_dict in guesses:
        DatasetUtils.pylot.imshow(guess_dict["img_array"])
        DatasetUtils.pylot.title(f"Guess: {categories[guess_dict["highest_guess"]]}")
        DatasetUtils.pylot.show()

# Menu Implementation

train_model_option = MenuOption(train_model_menu, "Train Model", condition_loaded_model)

predict_images_in_dir_menu = Menu(3, "Predict images in a Directory", [])
predict_images_in_dir_menu.update_options = update_predict_images_menu
predict_images_in_dir_menu.call_custom_function = predict_images_in_dir


#predict_single_image_menu = Menu(4, "Predict a single Image", [])

predict_images_menu = Menu(2, "Predict Images", [MenuOption(predict_single_image, "Predict a Single Image"), MenuOption(predict_images_in_dir_menu, "Predict image in Directory")])


save_model_option = MenuOption(save_model_menu, "Save Model", condition_loaded_model)

load_model_menu = Menu(1, "Load model menu", [])
load_model_menu.update_options = update_load_model_menu
load_model_menu.call_custom_function = load_model


main_menu = Menu(0, "Main menu", [
    MenuOption(load_model_menu), 
    save_model_option, 
    MenuOption(predict_images_menu, "Predict Images", condition_loaded_model),
    train_model_option
    ])

main_menu.update_title = update_main_menu_title
main_menu.exit_option_text = "Quit"

menu_handler = MenuHandler(main_menu)

menu_handler.main_loop()