# group_faces.py
import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from mtcnn import MTCNN

def group_faces(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Load the FaceNet model
    facenet_model = load_model('facenet_keras.h5')
    mtcnn = MTCNN()

    # Initialize database
    faces_db = []
    labels_db = []
    label_names = {}
    next_label = 0

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        if not img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping unreadable file '{img_name}'.")
            continue

        # Detect faces using MTCNN
        detections = mtcnn.detect_faces(image)

        if not detections:
            continue

        person_labels = set()
        for detection in detections:
            x, y, w, h = detection['box']
            x, y = max(x, 0), max(y, 0)  # Ensure coordinates are non-negative
            face = image[y:y+h, x:x+w]

            # Preprocess face for FaceNet
            face_resized = cv2.resize(face, (160, 160))
            face_resized = face_resized.astype('float32') / 255.0
            face_expanded = np.expand_dims(face_resized, axis=0)

            # Get face embedding
            face_embedding = facenet_model.predict(face_expanded)[0]

            # Match with database
            matched_label = None
            for idx, db_embedding in enumerate(faces_db):
                distance = np.linalg.norm(face_embedding - db_embedding)
                if distance < 0.5:  # Adjust threshold as needed
                    matched_label = labels_db[idx]
                    break

            if matched_label is None:
                matched_label = next_label
                folder_name = f"person_{next_label}"
                label_names[next_label] = folder_name
                next_label += 1

                faces_db.append(face_embedding)
                labels_db.append(matched_label)
                os.makedirs(os.path.join(output_folder, folder_name), exist_ok=True)

            person_labels.add(matched_label)

        # Save image in all matched person folders
        for label in person_labels:
            shutil.copy(img_path, os.path.join(output_folder, label_names[label]))

    print(f"Images have been grouped successfully in '{output_folder}'.")

if __name__ == '__main__':
    # Example usage
    input_folder = 'path_to_input_folder'
    output_folder = 'path_to_output_folder'
    group_faces(input_folder, output_folder)
