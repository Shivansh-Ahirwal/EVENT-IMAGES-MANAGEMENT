import os
import pickle
import shutil
from face_recognition import load_image_file, face_encodings, face_locations
from django.core.management.base import BaseCommand
import face_recognition

class Command(BaseCommand):
    help = "Group images by unique faces using face-recognition library."

    def add_arguments(self, parser):
        parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
        parser.add_argument('--output_dir', type=str, default='grouped_faces', help='Directory to store grouped images.')
        parser.add_argument('--encoding_file', type=str, default='face_encodings.dat', help='File to store face encodings.')
        parser.add_argument('--threshold', type=float, default=0.6, help='Face recognition threshold for matching.')

    def handle(self, *args, **options):
        image_dir = options['image_dir']
        output_dir = options['output_dir']
        encoding_file = options['encoding_file']
        threshold = options['threshold']

        # Initialize or load face encodings
        face_data = self.load_encodings(encoding_file)

        # Process each image
        self.process_images(image_dir, output_dir, encoding_file, threshold, face_data)

    def load_encodings(self, encoding_file):
        """
        Load existing face encodings from the file or initialize a new one.
        """
        if os.path.exists(encoding_file):
            with open(encoding_file, 'rb') as file:
                return pickle.load(file)
        return {}

    def save_encodings(self, encoding_file, face_data):
        """
        Save updated face encodings to the file.
        """
        with open(encoding_file, 'wb') as file:
            pickle.dump(face_data, file)

    def process_images(self, image_dir, output_dir, encoding_file, threshold, face_data):
        """
        Process all images in the directory, group them by detected faces.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            try:
                # Load image and detect faces
                image = load_image_file(image_path)
                face_locations_list = face_locations(image)
                face_encodings_list = face_encodings(image, face_locations_list)

                if not face_encodings_list:
                    self.stdout.write(f"No faces detected in {image_name}.")
                    continue

                for face_encoding in face_encodings_list:
                    # Check if the face matches existing encodings
                    matched = False
                    for person_id, stored_encoding in face_data.items():
                        matches = face_recognition.compare_faces(
                            [stored_encoding], face_encoding, tolerance=threshold
                        )
                        if matches[0]:  # Match found
                            matched = True
                            person_folder = os.path.join(output_dir, person_id)
                            break

                    if not matched:
                        # New face, assign a new ID and create a folder
                        person_id = f"Person_{len(face_data) + 1}"
                        person_folder = os.path.join(output_dir, person_id)
                        os.makedirs(person_folder)
                        face_data[person_id] = face_encoding

                    # Copy the image to the matched or newly created folder
                    shutil.copy(image_path, person_folder)

                self.stdout.write(f"Image {image_name} processed.")

            except Exception as e:
                self.stdout.write(f"Error processing image {image_name}: {str(e)}")

        # Save updated encodings
        self.save_encodings(encoding_file, face_data)
        self.stdout.write("Face grouping completed.")
