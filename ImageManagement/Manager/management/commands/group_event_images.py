import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
from face_recognition import load_image_file, face_encodings, face_locations
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Group images by unique faces using face-recognition library."

    def add_arguments(self, parser):
        parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
        parser.add_argument('--output_dir', type=str, default='grouped_faces', help='Directory to store grouped images.')
        parser.add_argument('--encoding_file', type=str, default='face_encodings.dat', help='File to store face encodings.')
        parser.add_argument('--threshold', type=float, default=0.6, help='Face recognition threshold for matching.')
        parser.add_argument('--workers', type=int, default=4, help='Number of threads for parallel processing.')

    def handle(self, *args, **options):
        image_dir = options['image_dir']
        output_dir = options['output_dir']
        encoding_file = options['encoding_file']
        threshold = options['threshold']
        num_workers = options['workers']

        # Initialize or load face encodings
        face_data = self.load_encodings(encoding_file)
        encodings_list = list(face_data.values())
        ids_list = list(face_data.keys())

        # Use NearestNeighbors for fast searching
        if encodings_list:
            nn_model = NearestNeighbors(metric="euclidean", algorithm="auto").fit(encodings_list)
        else:
            nn_model = None

        # Process images in parallel
        self.process_images(image_dir, output_dir, encoding_file, threshold, face_data, nn_model, ids_list, num_workers)

    def load_encodings(self, encoding_file):
        if os.path.exists(encoding_file):
            with open(encoding_file, 'rb') as file:
                return pickle.load(file)
        return {}

    def save_encodings(self, encoding_file, face_data):
        with open(encoding_file, 'wb') as file:
            pickle.dump(face_data, file)

    def process_image(self, image_path, threshold, nn_model, face_data, ids_list, output_dir):
        image_name = os.path.basename(image_path)
        try:
            # Load image and detect faces
            image = load_image_file(image_path)
            face_locations_list = face_locations(image)
            face_encodings_list = face_encodings(image, face_locations_list)

            if not face_encodings_list:
                return f"No faces detected in {image_name}."

            for face_encoding in face_encodings_list:
                matched = False
                if nn_model:
                    distances, indices = nn_model.kneighbors([face_encoding], n_neighbors=1)
                    if distances[0][0] <= threshold:
                        matched = True
                        person_id = ids_list[indices[0][0]]
                        person_folder = os.path.join(output_dir, person_id)

                if not matched:
                    # New face, assign a new ID
                    person_id = f"Person_{len(face_data) + 1}"
                    person_folder = os.path.join(output_dir, person_id)
                    os.makedirs(person_folder, exist_ok=True)
                    face_data[person_id] = face_encoding
                    ids_list.append(person_id)
                    if nn_model:
                        nn_model.fit(list(face_data.values()))

                # Copy the image to the folder
                shutil.copy(image_path, person_folder)

            return f"Image {image_name} processed."

        except Exception as e:
            return f"Error processing image {image_name}: {str(e)}"

    def process_images(self, image_dir, output_dir, encoding_file, threshold, face_data, nn_model, ids_list, num_workers):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png'))]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(
                lambda img_path: self.process_image(
                    img_path, threshold, nn_model, face_data, ids_list, output_dir
                ),
                image_paths,
            )

        for result in results:
            self.stdout.write(result)

        # Save updated encodings
        self.save_encodings(encoding_file, face_data)
        self.stdout.write("Face grouping completed.")
