import os
import shutil
from pathlib import Path

def copy_latest_models():
    # Get the current working directory
    current_dir = Path.cwd()
    source_dir = current_dir
    dest_dir = current_dir / "latest_model"

    print(f"Source Directory: {source_dir}")
    print(f"Destination Directory: {dest_dir}")

    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(exist_ok=True)
    print(f"Ensured that destination directory '{dest_dir}' exists.\n")

    # Iterate over each item in the source directory
    for model_name in os.listdir(source_dir):
        model_path = source_dir / model_name

        # Check if the item is a directory (i.e., a model directory)
        if model_path.is_dir() and model_name != "latest_model":
            print(f"Processing model directory: '{model_name}'")

            # Get all subdirectories within the model directory
            subdirs = [subdir for subdir in model_path.iterdir() if subdir.is_dir()]

            if not subdirs:
                print(f"  - No subdirectories found in '{model_name}'. Skipping.\n")
                continue

            # Find the latest subdirectory based on modification time
            latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
            print(f"  - Latest subdirectory: '{latest_subdir.name}' (Last Modified: {latest_subdir.stat().st_mtime})")

            # Define the destination path for the model's latest subdirectory
            dest_model_dir = dest_dir / model_name
            dest_model_dir.mkdir(parents=True, exist_ok=True)

            # Define the full destination path for the latest subdirectory
            dest_subdir_path = dest_model_dir / latest_subdir.name

            # Copy the latest subdirectory to the destination directory
            try:
                shutil.copytree(latest_subdir, dest_subdir_path)
                print(f"  - Copied '{latest_subdir.name}' to '{dest_model_dir}'.\n")
            except FileExistsError:
                print(f"  - Destination subdirectory '{dest_subdir_path}' already exists. Skipping copy.\n")
            except Exception as e:
                print(f"  - Error copying '{latest_subdir.name}': {e}\n")

    print("All applicable latest models have been copied successfully.")

if __name__ == "__main__":
    copy_latest_models()
