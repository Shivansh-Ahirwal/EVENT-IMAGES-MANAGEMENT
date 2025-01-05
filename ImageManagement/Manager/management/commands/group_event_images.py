import os
import pickle
import shutil
from face_recognition import load_image_file, face_encodings, face_locations, compare_faces
from django.core.management.base import BaseCommand


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

        face_data = self.load_encodings(encoding_file)

        os.makedirs(output_dir, exist_ok=True)
        unknown_folder = os.path.join(output_dir, 'unknown')
        os.makedirs(unknown_folder, exist_ok=True)

        self.process_images(image_dir, output_dir, unknown_folder, encoding_file, threshold, face_data)

    def load_encodings(self, encoding_file):
        """Load existing face encodings or initialize a new dictionary."""
        if os.path.exists(encoding_file):
            with open(encoding_file, 'rb') as file:
                return pickle.load(file)
        return {}

    def save_encodings(self, encoding_file, face_data):
        """Save updated face encodings to the file."""
        with open(encoding_file, 'wb') as file:
            pickle.dump(face_data, file)

    def process_images(self, image_dir, output_dir, unknown_folder, encoding_file, threshold, face_data):
        """Process images and group them by detected faces."""
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            try:
                image = load_image_file(image_path)
                face_locations_list = face_locations(image, model="cnn")  # Faster model: "hog" instead of "cnn"
                face_encodings_list = face_encodings(image, face_locations_list)

                if not face_encodings_list:
                    shutil.copy(image_path, unknown_folder)
                    self.stdout.write(f"No faces detected in {image_name}. Saved to 'unknown' folder.")
                    continue

                for face_encoding in face_encodings_list:
                    person_id, matched = self.find_or_create_match(face_encoding, face_data, threshold)

                    person_folder = os.path.join(output_dir, person_id)
                    os.makedirs(person_folder, exist_ok=True)

                    shutil.copy(image_path, person_folder)

                self.stdout.write(f"Image {image_name} processed.")

            except Exception as e:
                self.stdout.write(f"Error processing image {image_name}: {str(e)}")

        self.save_encodings(encoding_file, face_data)
        self.stdout.write("Face grouping completed.")

    def find_or_create_match(self, face_encoding, face_data, threshold):
        """Find a match for the face encoding or create a new entry."""
        for person_id, stored_encoding in face_data.items():
            matches = compare_faces([stored_encoding], face_encoding, tolerance=threshold)
            if matches[0]:
                return person_id, True

        new_person_id = f"Person_{len(face_data) + 1}"
        face_data[new_person_id] = face_encoding
        return new_person_id, False
    
