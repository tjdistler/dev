# Examine all image files in a directory and sort them into folders based on the 
# year they were taken. Use the EXIF data to determine the year. If EXIF data is 
# not available, then copy the image into an 'Unknown' folder for manual 
# review. A CSV file will be created in the 'Unknown' folder with the list of files
# in the 'Unknown' directory, plus a guess as to the year, and the names of the
# previous and next images (sorted alphabetically).
#
# Usage: python sort_by_year.py <source_dir> <target_dir>

import sys
import os
import csv
import shutil
import utils

UNKNOWN_DIR = "Unknown"

# Look at the surrounding years to guess the year of the current image.
def guess_year(prev_file, next_file):
    if prev_file is None or next_file is None:
        return None
    
    prev_year = utils.get_exif_creation_year(prev_file)
    next_year = utils.get_exif_creation_year(next_file)
    if prev_year == next_year:
        return prev_year
    return None

# Copy a file to a directory, creating the directory if it doesn't exist.
def copy_file_to_directory(file, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.basename(file)
    shutil.copyfile(file, os.path.join(directory, filename))


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
            creation_year = utils.get_exif_creation_year(full_path)
        
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
