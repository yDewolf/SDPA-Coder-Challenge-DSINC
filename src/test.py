import utils.DatasetUtils as DatasetUtils
import utils.ModelUtils as ModelUtils
import os

use_gray_scale: bool = False

# Create training data
X, y = DatasetUtils.compile_training_data("catDog_data", ["cat", "dog"], 32, use_gray_scale)
DatasetUtils.pickle_save_training_data("CDX", "CDy", X, y)
# X = X/255.0

# Load training data
# X, y = DatasetUtils.pickle_load_training_data("CDX.pickle", "DCy.pickle")

# Create and save
model = ModelUtils.create_model_convolutional(3, [64, 32], [(3, 3), (2, 2)], [(2, 2), (2, 2)], X.shape[1:])
# model.save("dog_cat.keras")


# Load
# model = ModelUtils.tensorflow.keras.models.load_model("dog_cat.keras")

# Train

ModelUtils.train_model_convolutional(model, X, y, 5)
model.save("dog_cat.keras")

# Predict

image_paths = []
for img_path in os.listdir("catDog_data/validate"):
    image_paths.append("catDog_data/validate/" + img_path)


categories = ["Cat", "Dog"]
guesses = ModelUtils.model_predict(model, categories, image_paths, convert_to_grayscale=use_gray_scale)
for guess_dict in guesses:
    DatasetUtils.pylot.imshow(guess_dict["img_array"])
    DatasetUtils.pylot.title(f"Guess: {categories[round(guess_dict["highest_guess"])]}")
    DatasetUtils.pylot.show()
