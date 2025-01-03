# management/commands/group_images.py
import subprocess
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Efficiently group images from an event into folders by detected faces using FaceNet."

    def add_arguments(self, parser):
        parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder containing event images.')
        parser.add_argument('--output-folder', type=str, required=True, help='Path to the folder where grouped images will be stored.')

    def handle(self, *args, **options):
        input_folder = options['input_folder']
        output_folder = options['output_folder']

        # Construct the command to run the script
        command = [
            'python',
            'scripts/group_faces.py',  # Path to the script
            '--input-folder', input_folder,
            '--output-folder', output_folder
        ]

        try:
            subprocess.run(command, check=True)
            self.stdout.write(self.style.SUCCESS(f"Images have been grouped successfully in '{output_folder}'."))
        except subprocess.CalledProcessError as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {e}"))
