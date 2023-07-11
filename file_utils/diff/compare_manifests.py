# Compares 2 manifest files for equality.
#
# Usage: python compare_manifests.py <manifest1> <manifest2>

import datetime
import sys
import manifesto

start_time = datetime.datetime.now()

if len(sys.argv) != 3:
    print("Usage: python compare_manifests.py <manifest1> <manifest2>")
    exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]

# Do the bulk of the work
file1_manifest = manifesto.load(file1)
file2_manifest = manifesto.load(file2)

identical = manifesto.compare(file1, file1_manifest, file2, file2_manifest)

print("Done. Total duration: " + str(datetime.datetime.now() - start_time))

if not identical:
    print("FAIL: Manifests are not identical")
    sys.exit(1)

print("PASS: Manifests are identical")