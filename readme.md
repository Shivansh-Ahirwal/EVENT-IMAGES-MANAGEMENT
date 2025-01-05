# Image Management Project

## Overview

This project automatically groups images by detecting unique faces using the `face_recognition` library. You can process images via a **Django management command** and then use a **web interface** to view the grouped results.

---

## Features

1. Detect faces in images.
2. Automatically group images by the same face.
3. Save unidentified faces in a separate "Unknown" folder.
4. View grouped images using a web interface.

---

## Requirements

- **Python 3.x**
- **Django** (Web Framework)
- **face_recognition** (Face detection and recognition)
- **scikit-learn** (Used for optimizations in image processing)

---

## Installation

1. **Clone the Repository**  
   Clone the project from GitHub:
   ```sh
   git clone https://github.com/Shivansh-Ahirwal/EVENT-IMAGES-MANAGEMENT
   cd EVENT-IMAGES-MANAGEMENT
   ```

2. **Set Up Virtual Environment**  
   Create and activate a virtual environment:
   ```sh
   python -m venv env
   source env/bin/activate   # For Linux/Mac
   env\Scripts\activate      # For Windows
   ```

3. **Install Dependencies**  
   Install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```

4. **Navigate to the Django Project Directory**  
   Move into the project folder:
   ```sh
   cd ImageManagement/Manager
   ```

---

## Workflow

### 1. Process Images (Run the Management Command)
Use the management command to process images and group them by faces. 

```sh
python manage.py group_event_images --image_dir <path_to_images> --output_dir <path_to_output> --encoding_file <path_to_encoding_file> --threshold <matching_threshold>
```

**Arguments**:
- `--image_dir`: Path to the folder containing your images.
- `--output_dir`: Path to save the grouped images.
- `--encoding_file`: File to save/load face encodings (useful for re-runs).
- `--threshold`: Matching sensitivity (default is `0.6`).

**Example**:
```sh
python manage.py group_event_images --image_dir ./images --output_dir ./grouped_images --encoding_file ./face_encodings.dat --threshold 0.6
```

After processing, your `output_dir` will have subfolders like:
```
grouped_images/
    Person_1/
    Person_2/
    ...
    unknown/
```
- Each folder contains images grouped by face.
- Images without any detectable faces will be in the `unknown` folder.

---

### 2. Start the Web Server (View Results in Browser)

Once the images are grouped, you can use the web interface to view the results:

1. Run the Django development server:
   ```sh
   python manage.py runserver
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

3. The homepage will display the grouped images with their respective folders.

---

## Example Workflow

1. **Prepare Images**  
   Place all event images in a folder (`Manager/Event-Images`).

2. **Group Images by Faces**  
   Run the management command:
   ```sh
   python manage.py group_event_images --image_dir Manager/Event-Images --output_dir Manager/Output-Images --encoding_file ./face_encodings.dat
   ```

3. **View Results**  
   Start the Django web server and view grouped images in the browser:
   ```sh
   python manage.py runserver
   ```

4. **Navigate to the Web Interface**  
   Go to `http://127.0.0.1:8000/` to explore the organized images.

---

## Folder Structure

```
EVENT-IMAGES-MANAGEMENT/
│
├── ImageManagement/
│   ├── Manager/              # Django project files
│   │   ├── management/
│   │   │   └── commands/     # Custom management commands
│   │   │       └── group_event_images.py
│   │   ├── settings.py       # Django settings
│   │   └── ...
│   ├── templates/            # HTML templates for web interface
│   └── ...
│
├── images/                   # Input images (to be processed)
├── grouped_images/           # Output folder for grouped images
├── face_encodings.dat        # File to store face encodings
└── requirements.txt          # Python dependencies
```

---

## Notes

- **Re-running the Command**: When re-running the command, the script uses `face_encodings.dat` to avoid recalculating encodings for already processed faces.
- **Threshold**: Adjust the `--threshold` argument to fine-tune face matching. Lower values are stricter, while higher values allow more flexible matching.
- **Unknown Faces**: Images without recognizable faces are stored in the `unknown` folder for manual review.

---
