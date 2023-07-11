# Creates a manifest of all files in a directory. The output is a CSV file that contains the path
# to each file and the SHA1 hash. A hash file of the manifist and a README are also created.
#
# Usage:
# python create_manifest.py <directory> <manifest>

import datetime
import os
import sys
import hashlib
import manifesto

start_time = datetime.datetime.now()

if len(sys.argv) != 3:
    print("Usage: python create_manifest.py <directory> <manifest>")
    exit(1)

directory = sys.argv[1]
manifest_fullpath = sys.argv[2]
manifest_dirname = os.path.dirname(manifest_fullpath)
manifest_filename = os.path.basename(manifest_fullpath)

# Make sure the directory is valid.
if not os.path.isdir(directory):
    print("Invalid directory: " + directory)
    sys.exit(1)

# Do the bulk of the work
manifest = manifesto.create(directory)
manifesto.write(manifest_fullpath, manifest)

# Create a SHA1 hash of the manifest file
hash_fullpath = manifest_fullpath + ".sha1"
with open(hash_fullpath, "w") as f:
    hash = hashlib.sha1()
    hash.update(open(manifest_fullpath, "rb").read())
    f.write(hash.hexdigest() + "  " + manifest_filename + "\n")

# Write README.txt file
end_time = datetime.datetime.now()
hash_filename = os.path.basename(hash_fullpath)
with open(os.path.join(manifest_dirname, "README.md"), "w") as f:
    f.write("RUN SUMMARY:\n")
    f.write("  Start Time:         " + str(start_time) + "\n")
    f.write("  End Time:           " + str(end_time) + "\n")
    f.write("  Command Line:       " + " ".join(sys.argv) + "\n")
    f.write("  Root Directory:     " + directory + "\n")
    f.write("  Files Processed:    " + str(len(manifest)) + "\n")
    f.write("  Execution Time:     " + str(end_time - start_time) + "\n")
    f.write("  Manifest:           " + manifest_filename + "\n")
    f.write("  Manifest Hash:      " + hash.hexdigest() + "\n")
    f.write("  Manifest Hash File: " + hash_filename + "\n")
    f.write("\n\n")
    f.write("USAGE:\n")
    f.write(" 1. Verify the manifest integrity: `shasum -c " + hash_filename + "`\n")
    f.write(" 2. Verify the contents on-disk against the manifest: `python verify_disk_contents.py <path to root directory> " + manifest_filename + "`\n")
    f.write(" 3. Compare the manifests between 2 drives: `python compare_manifests.py <manifest 1> <manifest 2>`\n")
    f.write("\n\n")
    f.write("- For copying between drives, use `cp -RpPXv <source> <target>`\n")
    f.write("- To delete all hidden files, use `find <directory> -type f -name '.*' -delete`\n")
    f.write("- To ZIP the contents of a directory, first `cd` to that directory, then use `zip -rT <zip file name> <directory>`\n")

print("Done. Total duration: " + str(end_time - start_time))