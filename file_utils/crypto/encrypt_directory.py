# Walks a directory and encrypts each file using AES GCM (ASE128, 96-bit nonce for NIST SP 800-38D compatibility).
# The key will be prompted for when the script is run. After a file is encrypted, the plaintext file will be deleted.
# EVERY file is encrypted regardless of if it has the .aes extension or not. This is to prevent skipping unecrypted
# files that erroneously have the .aes extension.
#
# Usage: python encrypt_directory.py <path>

import sys
import os
from encrypt_file import encrypt_file
import utils

if len(sys.argv) != 2:
    print("Usage: python encrypt_directory.py <path>")
    exit(1)

path = sys.argv[1]

# Verify path is valid
if not os.path.isdir(path):
    print("Path is not a directory")
    exit(1)

key = utils.prompt_for_key()
print(" Setting up key...")
key_obj = utils.Key(key)

# Walk the directory and encrypt each file
for root, dirs, files in os.walk(path):
    for filename in files:
        filepath = os.path.join(root, filename)
        encrypt_file(key_obj, filepath, filepath + ".aes")
        os.remove(filepath)