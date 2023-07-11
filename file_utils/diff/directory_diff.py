# Output the difference (at the directory level, not individual files) between 2 manifest files.
#
# Usage: python directory_diff.py <file1> <file2>

import datetime
import sys
import os
import manifesto

start_time = datetime.datetime.now()

if len(sys.argv) != 3:
    print("Usage: python directory_diff.py <file1> <file2>")
    exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

# Do the bulk of the work
file1_manifest = manifesto.load(file1)
file2_manifest = manifesto.load(file2)

# Iterate over manifest 1 and build a dictionary of directory paths that aren't in manifest 2.
dirs_in_1_not_2 = {} # Format: {path: file_count}
total_files_in_1 = {} # Format: {path: file_count}
for file in file1_manifest:
    path = os.path.dirname(file)
    if file not in file2_manifest:
        # print(" (1) -> " + file + " (" + path + ")")
        dirs_in_1_not_2[path] = dirs_in_1_not_2.get(path, 0) + 1
    total_files_in_1[path] = total_files_in_1.get(path, 0) + 1

# Iterate over manifest 2 and build a dictionary of directory paths that aren't in manifest 1.
dirs_in_2_not_1 = {} # Format: {path: file_count}
total_files_in_2 = {} # Format: {path: file_count}
for file in file2_manifest:
    path = os.path.dirname(file)
    if file not in file1_manifest:
        # print(" (2) -> " + file + " (" + path + ")")
        dirs_in_2_not_1[path] = dirs_in_2_not_1.get(path, 0) + 1
    total_files_in_2[path] = total_files_in_2.get(path, 0) + 1

print("Done. Total duration: " + str(datetime.datetime.now() - start_time))

for path in dirs_in_1_not_2:
    missing = dirs_in_1_not_2[path]
    total = total_files_in_1[path]
    suffix = ""
    if missing != total:
        suffix = " <--"
    print(" + . " + path + "\t(" + str(missing) + "/" + str(total) + " files)" + suffix)
for path in dirs_in_2_not_1:
    missing = dirs_in_2_not_1[path]
    total = total_files_in_2[path]
    suffix = ""
    if missing != total:
        suffix = " <--"
    print(" . + " + path + "\t(" + str(missing) + "/" + str(total) + " files)" + suffix)

if len(dirs_in_1_not_2) == 0 and len(dirs_in_2_not_1) == 0:
    print("No differences found.")