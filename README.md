
<h1 align="center">PoseSync: Flower Federated Learning for Yoga Pose Classification</h1>
<p align="center">
  <img src="https://github.com/Manoj-2702/Yoga-Pose-Estimation-by-Flower-Federated-Learning/assets/103581128/73bfd72c-120f-41fc-97e3-d277b4afcd4d" alt="Logo" />
</p>

This project demonstrates the use of Federated Learning with Flower for training a convolutional neural network (CNN) on a yoga pose classification task. The model is trained on distributed datasets across multiple clients, allowing for collaborative learning while preserving data privacy.

## Prerequisites

- Python 3.x
- TensorFlow
- Flower
- Scikit-learn
- Pillow
- Streamlit (for running the demo)

- `data/Resized_images1/`: Contains the training data for clients, organized into class folders.
- `data/ImageFolder/`: Contains test images for the demo.
- `client1.py` and `client2.py`: Flower client scripts that load and preprocess the data, define the model architecture, and participate in the federated learning process.
- `server.py`: Flower server script that coordinates the federated learning process and saves the aggregated model weights after each round.
- `test.py`: A Streamlit app for testing the trained model on new images.
- `README.md`: This file.


## 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start the Flower server

```bash
python server.py <server_port>
```
Replace <server_port> with the desired port number for the server.

#### 3. Start the Flower clients: 
Open two separate terminal windows and run the following commands in each window:

```bash
python client1.py <server_port>
python client2.py <server_port>
```
Replace <server_port> with the same port number used for the server. This will start the federated learning process, with the clients training the model on their local data and sending updates to the server after each round.

#### 4. Test the trained model: 
After the federated learning process is complete, you can test the trained model using the Streamlit app:

```bash
streamlit run test.py
```

This will open a web interface where you can upload an image, and the app will classify the yoga pose.


## Customization
- To use your own dataset, replace the data/Resized_images1/ folder with your own data, organized into class folders.
- Modify the client1.py and client2.py scripts to adjust the model architecture, hyperparameters, or data preprocessing steps as needed.
- Customize the server.py script to change the federated learning strategy or other server configurations.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
