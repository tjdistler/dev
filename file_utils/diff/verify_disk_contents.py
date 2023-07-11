# Verifies that the contents of a directory are identical to a manifest CSV file.
#
# Usage:
# python verify_disk_contents.py <directory> <manifest>

import datetime
import os
import sys
import manifesto

start_time = datetime.datetime.now()

if len(sys.argv) != 3:
    print("Usage: python create_manifest.py <directory> <manifest_filename>")
    exit(1)

directory = sys.argv[1]
manifest_filename = sys.argv[2]

# Make sure the directory is valid.
if not os.path.isdir(directory):
    print("Invalid directory: " + directory)
    sys.exit(1)

# Do the bulk of the work
file_manifest = manifesto.load(manifest_filename)
directory_manifest = manifesto.create(directory)

identical = manifesto.compare(manifest_filename, file_manifest, directory, directory_manifest)

end_time = datetime.datetime.now()

print("Done. Total duration: " + str(end_time - start_time))

if not identical:
    sys.exit(1)