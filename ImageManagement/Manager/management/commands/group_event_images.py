import os
import shutil
import numpy as np
from deepface import DeepFace
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Group images by unique faces using VGG-Face model."

    def add_arguments(self, parser):
        parser.add_argument(
            "--image_dir",
            type=str,
            required=True,
            help="Path to the directory containing event images.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="grouped_faces/",
            help="Path to save grouped images. Defaults to 'grouped_faces/'.",
        )
        parser.add_argument(
            "--distance_metric",
            type=str,
            default="cosine",
            help="Distance metric for face comparison (e.g., cosine, euclidean, euclidean_l2).",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.4,
            help="Threshold for face similarity (default: 0.4). Lower is stricter.",
        )

    def handle(self, *args, **options):
        image_dir = options["image_dir"]
        output_dir = options["output_dir"]
        distance_metric = options["distance_metric"]
        threshold = options["threshold"]

        self.stdout.write(f"Processing images from {image_dir}...")
        self.group_images_by_unique_faces(image_dir, output_dir, distance_metric, threshold)
        self.stdout.write("Images have been grouped successfully!")

    def group_images_by_unique_faces(self, image_dir, output_dir, distance_metric, threshold):
        """
        Group images by unique faces using VGG-Face.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List of unique face embeddings and folder mappings
        unique_faces = []
        unique_face_folders = []

        for file in os.listdir(image_dir):
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(image_dir, file)

                try:
                    # Extract face embedding using VGG-Face model
                    face_embedding = DeepFace.represent(
                        img_path=image_path, model_name="VGG-Face", enforce_detection=False
                    )

                    # Compare face embedding with known unique faces
                    match_found = False
                    for idx, unique_embedding in enumerate(unique_faces):
                        # Manually calculate the cosine distance
                        distance = self.calculate_cosine_distance(
                            face_embedding[0]["embedding"], unique_embedding[0]["embedding"]
                        )

                        if distance <= threshold:
                            # Match found, assign to the same folder
                            match_found = True
                            folder_name = unique_face_folders[idx]
                            break

                    if not match_found:
                        # New unique face detected
                        folder_name = f"Person_{len(unique_faces) + 1}"
                        unique_faces.append(face_embedding)
                        unique_face_folders.append(folder_name)

                        # Create folder for the new unique face
                        os.makedirs(os.path.join(output_dir, folder_name))

                    # Copy the image to the appropriate folder
                    destination = os.path.join(output_dir, folder_name)
                    shutil.copy(image_path, destination)
                    self.stdout.write(f"Image {file} stored in {folder_name}")

                except Exception as e:
                    self.stderr.write(f"Error processing image {file}: {str(e)}")

    def calculate_cosine_distance(self, embedding1, embedding2):
        """
        Calculate cosine distance between two face embeddings.
        """
        # Convert embeddings to numpy arrays
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm_embedding1 = np.linalg.norm(embedding1)
        norm_embedding2 = np.linalg.norm(embedding2)

        # Cosine distance formula
        cosine_similarity = dot_product / (norm_embedding1 * norm_embedding2)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance
