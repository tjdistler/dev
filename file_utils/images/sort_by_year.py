# Examine all image files (recursively) in a directory and sort them into folders
# based on the year they were taken. Use the EXIF data to determine the year. If
# EXIF data is not available, then sort the image into an 'Unknown' folder for 
# manual review.
#
# Usage: python sort_by_year.py <source_dir> <target_dir>
#
# Requires exiftool (brew install exiftool)

import subprocess
import sys
import os
import shutil

UNKNOWN_DIR = "Unknown"

# Copy a file to a directory, creating the directory if it doesn't exist.
def copy_file_to_directory(file, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("-> Created directory: " + directory)
    filename = os.path.basename(file)
    shutil.copyfile(file, os.path.join(directory, filename))

# Uses the EXIF data to get the year the image/video was taken. Returns the year
# as a string or None if not available.
def get_exif_creation_year(file):
    EXIFTOOL_CREATION_DATE_TAG = "Creation Date"
    EXIFTOOL_CREATE_DATE_TAG = "Create Date"
    process = subprocess.Popen(["exiftool", file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    lines = out.decode("utf-8").split("\n")
    creation_year = None
    for line in lines:
        if line.startswith(EXIFTOOL_CREATION_DATE_TAG):
            creation_year = line.split(":")[1].strip() # Prefer this entry
        elif line.startswith(EXIFTOOL_CREATE_DATE_TAG) and creation_year is None:
            creation_year = line.split(":")[1].strip() # Fallback to this entry
    return creation_year


if len(sys.argv) < 3:
    print("Usage: python sort_by_year.py <source_dir> <target_dir>")
    exit(1)

source_dir = sys.argv[1]
target_dir = sys.argv[2]
unknown_dir = os.path.join(target_dir, UNKNOWN_DIR)

# Walk through all the files in the source directory.
for root, dirs, files in os.walk(source_dir):
    for file in files:
        full_path = os.path.join(root, file)
        try:
            creation_year = get_exif_creation_year(full_path)
    
            # If the EXIF data is not available, then move the file to the 'unknown' folder.
            if creation_year is None:
                copy_file_to_directory(full_path, unknown_dir)
            else:
                copy_file_to_directory(full_path, os.path.join(target_dir, creation_year))

            print("File: " + full_path + " | Year: " + str(creation_year))

        except Exception as e:
            print("Error: " + str(e))
            continue
