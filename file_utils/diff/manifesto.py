# Helper functions for working with manifests.

import csv
import datetime
import hashlib
import os
import sys

CHUNK_SIZE = 1024 * 1024 * 1024 # 1 GiB

# Build a manifest from the contents of a directory. Manifest format: {filename: hash}
def create(directory):
    print("Building manifest from directory: " + directory + "...")
    realpath = os.path.realpath(directory)
    manifest = _get_file_list(realpath)
    return _hash_files(realpath, manifest)


# Write a manifest to a file. Manifest format: {filename: hash}
def write(filename, files):
    print("Writing manifest to file: " + filename + "...")
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["path", "hash"])
        writer.writeheader()
        for file in files:
            writer.writerow({"path": file, "hash": files[file]})


# Load the contents of a manifest file. Manifest format: {filename: hash}
def load(filename):
    print("Loading manifest...")
    manifest = {}
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # Skip the header row
        for row in reader:
            manifest[row[0]] = row[1]
            _console_overwrite(" " + str(len(manifest)))
    print("\r   Loaded " + str(len(manifest)) + " files           ")
    return manifest


# Compares the contents of 2 manifests. Manifest format: {filename: hash}
# Returns True if the manifests are identical, False otherwise.
def compare(manifest1_dirname, manifest1, manifest2_dirname, manifest2):
    print("Comparing manifests...")
    if not _compare_directory_structure(manifest1_dirname, manifest1, manifest2_dirname, manifest2):
        return False
    
    print(" -> Comparing file hashes...")
    mismatch_count = 0
    for i,file in enumerate(manifest1):
        if manifest1[file] != manifest2[file]:
            mismatch_count += 1
            print("   - " + file)
    print("\r{} mismatched file(s) found                       ".format(mismatch_count))
    return mismatch_count == 0


# Recursively removes all instances of a file from a directory.
def remove_file_recursively(directory, target):
    print("Removing all instances of " + target + "...")
    total = 0;
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename == target:
                total += 1
                os.remove(os.path.join(root, filename))
    print("   Removed {} instances".format(total))


def _console_overwrite(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()
    return len(msg)


# A function that prints progress percentage to the console. Returns the length of the message.
def _print_progress(current, total, duration):
    msg = " {}/{} ({:.2f}%) - {}".format(current, total, current/total*100, duration)
    return _console_overwrite(msg)


# Recusively walks a directory and returns a dictionary of all files. Format {<file_path>: None}
def _get_file_list(directory):
    print(" -> Listing files...")
    all_files = {}
    files = []
    for root, dirs, files in os.walk(directory):
        # Remove directory prefix from start of root
        root = root[len(directory)+1:]
        for file in files:
            # Add file to dictionary but set the hash to None for now
            all_files[os.path.join(root, file)] = None
            _console_overwrite("   " + str(len(all_files)))
    print("\r   Found " + str(len(all_files)) + " files           ")
    return all_files;


# Takes a dictionary of filenames and calculates the hash for each. Format {<file_path>: <hash>}
def _hash_files(directory, files):
    print(" -> Hashing files...")
    start_time = datetime.datetime.now()
    total = len(files);
    for i,file in enumerate(files):
        _print_progress(i, total, datetime.datetime.now() - start_time)

        # Calculate the SHA-1 hash of the file.
        hash = hashlib.sha1()
        with open(os.path.join(directory, file), "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hash.update(chunk)
        files[file] = hash.hexdigest();
    print("\r   Hashed " + str(len(files)) + " files - Total duration: " + str(datetime.datetime.now() - start_time))
    return files;



# Compares the structural contents of two directories. Input dictionary format: {filename: hash}.
# Returns True if the contents are identical, False otherwise.
def _compare_directory_structure(dirname1, dirfiles1, dirname2, dirfiles2):
    print(" -> Comparing directory structures...")    
    start_time = datetime.datetime.now()
    dir1_msg_displayed = False
    dir2_msg_displayed = False
    dir1_total_mismatches = 0
    dir2_total_mismatches = 0

    total = len(dirfiles1)
    for i,key in enumerate(dirfiles1):
        if key not in dirfiles2:
            dir1_total_mismatches += 1

            # Print the name of mismatched file
            if not dir1_msg_displayed:
                print("\r  -> Files in " + dirname1 + " but not in " + dirname2)
                dir1_msg_displayed = True
            print("\r    - " + key + "                              ")
        _print_progress(i, total, datetime.datetime.now() - start_time)

    total = len(dirfiles2)
    for key in dirfiles2:
        if key not in dirfiles1:
            dir2_total_mismatches += 1

            # Print the name of mismatched file
            if not dir2_msg_displayed:
                print("\r  -> Files in " + dirname2 + " but not in " + dirname1)
                dir2_msg_displayed = True
            print("\r    - " + key + "                              ")
        _print_progress(i, total, datetime.datetime.now() - start_time)

    if dir1_total_mismatches != 0 and dir2_total_mismatches != 0:
        print("\r  -> Mismatches found:                         ")
        print("    - " + str(dir1_total_mismatches) + " in " + dirname1)
        print("    - " + str(dir2_total_mismatches) + " in " + dirname2)
    else:
        sys.stdout.write("\r                                                  \r")
    
    return dir1_total_mismatches == 0 and dir2_total_mismatches == 0
