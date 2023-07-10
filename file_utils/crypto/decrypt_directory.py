# Walks a directory and decrypts any file with the .aes extension using AES GCM (ASE128, 96-bit nonce for NIST SP 800-38D 
# compatibility). The key will be prompted for when the script is run. After a file is decrypted, the .aesd file will be 
# deleted.
#
# Usage: python decrypt_directory.py <path>

import sys
import os
from decrypt_file import decrypt_file
import utils

if len(sys.argv) != 2:
    print("Usage: python decrypt_directory.py <path>")
    exit(1)

path = sys.argv[1]

# Verify path is valid
if not os.path.isdir(path):
    print("Path is not a directory")
    exit(1)

key = utils.prompt_for_key()

# Walk the directory and encrypt each file
for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".aes"):
            filepath = os.path.join(root, filename)
            try:
                decrypt_file(filepath, os.path.splitext(filepath)[0], key=key)
            except Exception as e:
                print(" -> Failed to decrypt file: " + str(e))
                continue
            os.remove(filepath)