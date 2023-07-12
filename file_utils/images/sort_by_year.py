# Examine all image files in a directory and sort them into folders based on the 
# year they were taken. Use the EXIF data to determine the year. If EXIF data is 
# not available, then copy the image into an 'Unknown' folder for manual 
# review. A CSV file will be created in the 'Unknown' folder with the list of files
# in the 'Unknown' directory, plus a guess as to the year, and the names of the
# previous and next images (sorted alphabetically).
#
# Usage: python sort_by_year.py <source_dir> <target_dir>
#
# Requires exiftool (brew install exiftool)
# Requires Pillow (python3 -m pip install --upgrade Pillow)
# Requires Pillow-heif:
#   https://pypi.org/project/pillow-heif/
#   `brew install x265 libjpeg libde265 libheif`
#   `python3 -m pip install --upgrade pillow-heif --no-binary :all:`

import sys
import os
import csv
import shutil
import subprocess
from PIL import Image
from pillow_heif import HeifImagePlugin

UNKNOWN_DIR = "Unknown"

# Look at the surrounding years to guess the year of the current image.
def guess_year(prev_file, next_file):
    if prev_file is None or next_file is None:
        return None
    
    prev_year = get_exif_creation_year(prev_file)
    next_year = get_exif_creation_year(next_file)
    if prev_year == next_year:
        return prev_year
    return None

# Copy a file to a directory, creating the directory if it doesn't exist.
def copy_file_to_directory(file, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.basename(file)
    shutil.copyfile(file, os.path.join(directory, filename))

# Uses the EXIF data to get the year the image/video was taken. Returns the year
# as a string or None if not available.
def get_exif_creation_year(file):
    creation_year = None

    # Start by calling out to exiftool (most reliable and supporting)
    EXIFTOOL_CREATION_DATE_TAG = "Creation Date"
    EXIFTOOL_CREATE_DATE_TAG = "Create Date"
    process = subprocess.Popen(["exiftool", file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    lines = out.decode("utf-8").split("\n")
    for line in lines:
        if line.startswith(EXIFTOOL_CREATION_DATE_TAG):
            creation_year = line.split(":")[1].strip() # Prefer this entry
            break
        elif line.startswith(EXIFTOOL_CREATE_DATE_TAG) and creation_year is None:
            creation_year = line.split(":")[1].strip() # Fallback to this entry

    # If creation_year is still None, then try the Pillow library.
    if creation_year is None:
        image = Image.open(file)
        exif = image.getexif()
        if exif is not None:
            dt_original = exif.get(306)
            if dt_original is not None:
                creation_year = dt_original[0:4]
    
    return creation_year


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sort_by_year.py <source_dir> <target_dir>")
        exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    unknown_dir = os.path.join(target_dir, UNKNOWN_DIR)

    # List all the files in the source directory and sort them. iPhone image file names
    # have the form IMG_XXXX, so by sorting them we put images next to each other that
    # were taken sequenial to each other. This allows us to guess the year of a file
    # without EXIF data by looking at the files surrounding it.
    files = []
    for file in os.listdir(source_dir):
        # Skip file if it is a directory.
        full_path = os.path.join(source_dir, file)
        if os.path.isdir(full_path):
            continue
        files.append(file)

    files.sort()

    # A list-of-lists of files that could not be sorted for generating a CSV file.
    # Item format: [file, guess, prev_file, next_file]
    unknown_files = []
    for idx,file in enumerate(files):
        try:
            full_path = os.path.join(source_dir, file)
            creation_year = get_exif_creation_year(full_path)
        
            # If the EXIF data is not available, then attempt to guess the year by looking
            # at the files surrounding it.
            guess = None
            if creation_year is None:
                prev_file = None
                next_file = None
                if idx > 0:
                    prev_file = os.path.join(source_dir, files[idx - 1])
                if idx < len(files) - 1:
                    next_file = os.path.join(source_dir, files[idx + 1])
                guess = guess_year(prev_file, next_file)
                unknown_files.append([file, str(guess), os.path.basename(str(prev_file)), os.path.basename(str(next_file))])
                copy_file_to_directory(full_path, unknown_dir)
            else:
                copy_file_to_directory(full_path, os.path.join(target_dir, creation_year))

            print("({}/{}) {} | Year: {} {}".format(idx + 1, len(files), file, creation_year, 
                                                    ("(guess: " + str(guess) + ")") if guess is not None else ""))

        except Exception as e:
            print("* File: " + full_path + " | Error: " + str(e))
            continue
    
    # Write a CSV file of the unknown files to the "unknown" directory.
    if len(unknown_files) > 0:
        csv_filename = os.path.join(unknown_dir, "unknown_files.csv")
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["file", "guess", "prev_file", "next_file"])
            writer.writeheader()
            for entry in unknown_files:
                writer.writerow({"file": entry[0], "guess": entry[1], "prev_file": entry[2], "next_file": entry[3]})
        print("Unknown files metadata written to: " + csv_filename)
