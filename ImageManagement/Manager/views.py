import os
from django.shortcuts import render
from django.conf import settings

def display_grouped_faces(request):
    output_dir = os.path.join(settings.BASE_DIR, 'Manager/Output-Images')  # Adjusted to your folder
    images = []

    if os.path.exists(output_dir):
        for folder_name in os.listdir(output_dir):
            folder_path = os.path.join(output_dir, folder_name)
            if os.path.isdir(folder_path):
                # Get the first image from the folder
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_path = os.path.join(settings.MEDIA_URL, folder_name, file_name).replace('\\', '/')
                        images.append((folder_name, image_path))
                        break

    return render(request, 'display_faces.html', {'images': images})
