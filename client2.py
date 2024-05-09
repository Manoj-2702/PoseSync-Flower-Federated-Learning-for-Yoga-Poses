import flwr as fl
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import sys
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load and compile Keras model
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
model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)
print(model.summary())

def load_images_from_folder(folder, img_height=28, img_width=28):
    images = []
    labels = []
    label_encoder = LabelEncoder()  # Initialize label encoder
    all_labels = []  # Collect all labels to fit the encoder

    # First pass to collect labels
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            label = os.path.basename(subdir)  # Assuming folder names are labels
            all_labels.append(label)

    # Fit label encoder
    label_encoder.fit(all_labels)

    # Second pass to process images
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".png") or filepath.endswith(".jpg"):
                img = Image.open(filepath).convert('L')
                img = img.resize((img_width, img_height))
                images.append(np.array(img))
                label = os.path.basename(subdir)  # Assuming folder names are labels
                labels.append(label_encoder.transform([label])[0])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load your images
folder_path = './data/newImages'
images, labels = load_images_from_folder(folder_path)

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config=None):
        return model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        # model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(hist['accuracy'])
        plt.plot(hist['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(2, 2, 2)
        plt.plot(hist['loss'])
        plt.plot(hist['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        print("\n\n\n----------------  Test ----------------- ")
        # model.set_weights(parameters)
        model.set_weights(parameters)
        y_pred = model.predict(x_test)
        y_true = np.argmax(y_test, axis=1)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        num_classes = y_test.shape[1]
        y_true_binarized = label_binarize(y_true, classes=range(num_classes))
        y_pred_binarized = y_pred

        # Calculate ROC curve and area under the curve (AUC) for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'gray']
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)

