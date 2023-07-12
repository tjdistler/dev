# Examines the EXIF data in all files in a directory and verifies that the creation year
# matches a given year (command line argument) OR that the year is not specified. A CSV
# file is written listing all files with an incorrect creation year.
#
# Usage: python verify_year.py <year> <directory>

import os
import sys
import csv
import utils
import __main__

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python verify_year.py <year> <directory>')
        sys.exit(0)

    target_year = sys.argv[1]
    directory = sys.argv[2]

    if len(target_year) != 4:
        print('Error: year must be 4 digits long')
        sys.exit(0)

    if not os.path.isdir(directory):
        print('Error: directory does not exist')
        sys.exit(0)
    
    # Iterate over all files in the directory and check the creation year
    misplaced_files = [] # Format: list of lists [filename, target_year, actual_year]
    files = os.listdir(directory)
    for idx,file in enumerate(files):
        # Skip directories
        if os.path.isdir(os.path.join(directory, file)):
            continue
        try:
            sys.stdout.write("({}/{}) Checking {}".format(idx, file, target_year))
            year = utils.get_exif_creation_year(os.path.join(directory, file))
            if year is None or year == target_year:
                print()
                continue
            else:
                print(" <-- Incorrect year ({})".format(year))
                misplaced_files.append([file, target_year, year])
        except Exception as e:
            print(" <-- Error: {}".format(e))
    
    # Write misplaced files to a CSV file
    if len(misplaced_files) > 0:
        csv_filename = os.path.join(directory, "misplaced_files.csv")
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["file", "target_year", "actual_year"])
            writer.writeheader()
            for entry in misplaced_files:
                writer.writerow({"file": entry[0], "target_year": entry[1], "actual_year": entry[2]})
        print("Misplaced files metadata written to: " + csv_filename)