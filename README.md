Face Recognition using kNN with Hyperparameter Tuning
This project implements a face recognition system using the k-Nearest Neighbors (kNN) algorithm. The system detects faces in images, preprocesses them, applies dimensionality reduction using PCA, and uses hyperparameter tuning to optimize the kNN classifier. The project also includes real-time face recognition using a webcam feed.

Table of Contents
Introduction
Installation
Usage
Project Structure
Model Training
Real-Time Face Recognition
Contributing
License
Introduction
This face recognition system leverages OpenCV for face detection, PCA for dimensionality reduction, and the k-Nearest Neighbors (kNN) algorithm for classification. Hyperparameter tuning is performed using Grid Search with Cross-Validation to find the optimal parameters for the kNN model.

Installation
Clone the repository:

sh
Copy code
git clone https://github.com/yourusername/face-recognition-knn.git
cd face-recognition-knn
Install the required libraries:

sh
Copy code
pip install -r requirements.txt
Required Libraries:

OpenCV
NumPy
Scikit-learn
Imbalanced-learn
Usage
Prepare your dataset:

Organize your images in subdirectories named after the person in the images. Each subdirectory should contain images of that person.
Example:
Copy code
dataset/
  ├── person1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  ├── person2/
  │   ├── image1.jpg
  │   ├── image2.jpg
Update the data_path in the script to point to your dataset directory:

python
Copy code
data_path = r"C:\Users\prade\Downloads\cropped_images"
Run the script:

sh
Copy code
python face_recognition.py
Project Structure
face_recognition.py: Main script containing the code for face detection, dataset loading, model training, and real-time face recognition.
requirements.txt: List of required Python libraries.
Model Training
The script performs the following steps for training the model:

Load Dataset: Detects faces in images and resizes them.
Preprocess Data: Flattens images and standardizes the data.
Dimensionality Reduction: Applies PCA to reduce the dimensionality of the data.
Handle Class Imbalance: Uses RandomOverSampler to balance the dataset.
Hyperparameter Tuning: Performs Grid Search with Cross-Validation to find the best parameters for the kNN model.
Train kNN Model: Trains the kNN model with the optimal parameters.
Evaluate Model: Calculates the accuracy of the model on the test set.
Real-Time Face Recognition
The script includes code to capture video from a webcam, detect faces in the video frames, and classify them using the trained kNN model. The recognized person's name is displayed on the video feed.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
