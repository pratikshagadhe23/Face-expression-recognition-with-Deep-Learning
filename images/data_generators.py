from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Input path for the images
base_path = "/Users/pratiksha/Documents/Pratiksha/Documents/GitHub/GitHub/Face-expression-recognition-with-Deep-Learning/images/"
# Size of the image: 48x48 pixels
pic_size = 48

# Number of images to feed into the NN for every batch
batch_size = 128

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(
    directory=base_path + "train",
    target_size=(pic_size, pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = datagen_validation.flow_from_directory(python3 data_generators.py

    directory=base_path + "validation",
    target_size=(pic_size, pic_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Save the generators using pickle
with open('/Users/pratiksha/Documents/Pratiksha/Documents/GitHub/GitHub/Face-expression-recognition-with-Deep-Learning/train_generator.pkl', 'wb') as f:
    pickle.dump(train_generator, f)

with open('/Users/pratiksha/Documents/Pratiksha/Documents/GitHub/GitHub/Face-expression-recognition-with-Deep-Learning/validation_generator.pkl', 'wb') as f:
    pickle.dump(validation_generator, f)
