import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return gray[y : y + w, x : x + h]


def load_dataset(data_path):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in label_map:
            label_map[person_name] = label_id
            label_id += 1

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            face = detect_face(img)
            if face is not None:
                face = cv2.resize(face, (100, 100))
                images.append(face)
                labels.append(label_map[person_name])

    return np.array(images), np.array(labels), label_map


data_path = r"C:\Users\pradeep\Downloads\cropped_images"
images, labels, label_map = load_dataset(data_path)

print("Label Map:", label_map)
print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)

# Training Model
n_samples, height, width = images.shape
X = images.reshape((n_samples, height * width))
scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, labels, test_size=0.2, random_state=42
)

ros = RandomOverSampler()
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Hyperparameter tuning for kNN using Grid Search
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_res, y_train_res)

best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

knn = KNeighborsClassifier(**best_params)
knn.fit(X_train_res, y_train_res)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Predicting
def predict(face, knn, scaler, pca, label_map):
    if face is None:
        return "No face detected"

    face = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
    face = scaler.transform(face)
    face_pca = pca.transform(face)
    label_id = knn.predict(face_pca)[0]

    name_map = {v: k for k, v in label_map.items()}
    return name_map[label_id]


# OpenCV code to capture video and make predictions
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)
    if face is not None:
        label = predict(face, knn, scaler, pca, label_map)
        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        (x, y, w, h) = cv2.boundingRect(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
