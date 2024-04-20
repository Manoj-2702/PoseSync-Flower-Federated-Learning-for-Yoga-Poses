# import numpy as np
# import tensorflow as tf
# from PIL import Image

# def load_and_pad_weights(weights_path, model, padding_value=0.01):
#     data = np.load(weights_path, allow_pickle=True)
#     loaded_weights = [data[key] for key in sorted(data.keys())]
    
#     model_weights_shapes = [w.shape for layer in model.layers for w in layer.get_weights()]
#     padded_weights = []

#     idx = 0
#     for shape in model_weights_shapes:
#         if idx < len(loaded_weights) and loaded_weights[idx].shape == shape:
#             padded_weights.append(loaded_weights[idx])
#             idx += 1
#         else:
#             # Pad with a constant value or small random values instead of zeros
#             print(f"Padding required for layer with shape {shape}")
#             if len(shape) > 1:
#                 # Use random padding for weight matrices
#                 padding = np.random.normal(loc=padding_value, scale=0.05, size=shape)
#             else:
#                 # Use a constant padding for bias vectors
#                 padding = np.full(shape, padding_value)
            
#             # print(f"Padding values for shape {shape}: {padding}")
#             padded_weights.append(padding)

#     return padded_weights

# # Recreate the model architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)),
#     tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
#     tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=256, activation='relu'),
#     tf.keras.layers.Dense(units=164, activation='relu'),
#     tf.keras.layers.Dense(units=8, activation='softmax')
# ])
# weights_path = "round-90-weights.npz"
# padded_weights = load_and_pad_weights(weights_path, model)
# model.set_weights(padded_weights)

# # Load and preprocess the input image
# img_path = './data/ImageFolder/Veerabhadrasana_40.jpeg'  # Replace with the actual image path
# img = Image.open(img_path).convert('L')
# img = img.resize((28, 28))
# img = np.array(img).reshape(1, 28, 28, 1)

# # Classify the input image
# predictions = model.predict(img)
# predicted_class = np.argmax(predictions)
# print(f"Predicted class: {predicted_class}")

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_pad_weights(weights_path, model, padding_value=0.01):
    data = np.load(weights_path, allow_pickle=True)
    loaded_weights = [data[key] for key in sorted(data.keys())]
    model_weights_shapes = [w.shape for layer in model.layers for w in layer.get_weights()]
    padded_weights = []
    idx = 0
    for shape in model_weights_shapes:
        if idx < len(loaded_weights) and loaded_weights[idx].shape == shape:
            padded_weights.append(loaded_weights[idx])
            idx += 1
        else:
            if len(shape) > 1:
                padding = np.random.normal(loc=padding_value, scale=0.05, size=shape)
            else:
                padding = np.full(shape, padding_value)
            padded_weights.append(padding)
    return padded_weights

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=164, activation='relu'),
        tf.keras.layers.Dense(units=8, activation='softmax')
    ])
    return model

model = create_model()
weights_path = "round-20-weights.npz"  # Ensure this path is correct and accessible
padded_weights = load_and_pad_weights(weights_path, model)
model.set_weights(padded_weights)

def preprocess_image(image):
    img = Image.open(image).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    return img

def classify_image(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    return np.argmax(predictions)

st.title('Yoga Pose Classification')
st.title('Done with ❤️❤️')
st.title('Manoj, Shreya NP, Shreya Gunnan, Prerana')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = classify_image(uploaded_file)
    st.write(f'Predicted class: {label}')