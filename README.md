# Face Recognition System with KNN Classifier
This Python script implements a face recognition system using a K-Nearest Neighbors (KNN) classifier. It performs the following tasks:

Data Loading:Loads images from a directory structure where each sub-directory represents a person and contains their images.

Preprocessing: Converts images to grayscale, detects faces using a Haar cascade classifier, resizes faces to a fixed size, flattens the images, and applies standardization and dimensionality reduction techniques.

Class Imbalance Handling: Optionally uses Random Oversampling to address class imbalance if the number of images varies significantly per person.

Hyperparameter Tuning: Performs a grid search to find the best parameters for the KNN classifier.

Model Training: Trains the KNN classifier with the best parameters.

Evaluation: Evaluates the model's accuracy on a hold-out test set.

Prediction: Predicts the identity of a person in a new image.

Real-time Recognition: Utilizes OpenCV to capture video from a webcam and predict the identity of the person in each frame.


Requirements:
Python 3.x
OpenCV
NumPy
scikit-learn
imblearn
Instructions:

Install required libraries: Use pip install opencv-python numpy scikit-learn imblearn
Prepare Dataset: Create a directory structure where each sub-directory represents a person and contains their images.
Modify data_path: Update the data_path variable in the script to point to your dataset directory.
Run the script: Execute the script using python face_recognition.py
Explanation:

The script defines functions for loading the dataset, detecting faces, preprocessing images, training the KNN model, predicting identities, and performing real-time recognition using a webcam.
The load_dataset function parses the dataset directory, loads images, performs face detection, and creates a label map for each person.
The training process involves flattening images, applying standardization with StandardScaler, dimensionality reduction with PCA, performing a train-test split, handling class imbalance (optional), hyperparameter tuning with GridSearchCV, training the KNN model, and evaluating its accuracy.
The predict function takes a new face image, preprocesses it, and predicts the identity using the trained KNN model.
The OpenCV code continuously captures video frames, detects faces, performs prediction, and displays the predicted name on the frame.
This script provides a basic implementation of a face recognition system. You can further improve it by:

Experimenting with different face detection algorithms.
Using more sophisticated feature extraction techniques like Local Binary Patterns (LBP) or deep learning models.
Implementing a more robust person identification system that handles variations in lighting, pose, and occlusion.


