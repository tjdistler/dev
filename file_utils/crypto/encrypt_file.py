# Encrypts a file using AES GCM (ASE128, 96-bit nonce for NIST SP 800-38D compatibility).
# The key will be prompted for if this script is executed directly. The unencrypted file
# will be removed.
#
# Usage: python encrypt_file.py <file_name>
# Requires: pycryptodome (pip install pycryptodome)

import json
import os
import sys
from base64 import b64encode
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt
from Cryptodome.Random import get_random_bytes
import utils


# Encrypts infile and writes ciphertext to outfile. 'key' is a Key object.
# 
# The outfile format is:
# <json header>
# \n\n
# <binary ciphertext>
#
# JSON header:
# {
#   "version": 1,
#   "key_params":  {
#       "kdf": "scrypt",
#       "kdf_params": { "dklen": 32, "N": 1048576, "r": 8, "p": 1, "salt": <base64-encoded salt> }
#   },
#   "nonce": <base64-encoded nonce>,
#   "tag": <base64-encoded tag>
# }
#
def encrypt_file(key_obj, infile, outfile):

    # Encrypt the file using the derived key.
    with open(infile, "rb") as f:
        plaintext = f.read()
        nonce = get_random_bytes(12)
        cipher = AES.new(key_obj.dkey, AES.MODE_GCM, nonce=nonce)
        print(" Encrypting {}...".format(infile))
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    # Build the JSON header.
    header = {
        "version": 1,
        "key_params": {
            "kdf": "scrypt",
            "kdf_params": {
                "dklen": key_obj.dkey_len,
                "N": key_obj.N,
                "r": key_obj.r,
                "p": key_obj.p,
                "salt": b64encode(key_obj.salt).decode('utf-8')
         }
        },
        "nonce": b64encode(nonce).decode('utf-8'),
        "tag": b64encode(tag).decode('utf-8')
    }

    # Write output file
    with open(outfile, "wb") as f:
        f.write(json.dumps(header).encode('utf-8')) # Write header as a JSON object.
        f.write(b"\n\n") # Separate header from ciphertext.
        f.write(ciphertext)


if __name__ == "__main__":
    infile = sys.argv[1]
    key = utils.prompt_for_key()
    print(" Setting up key...")
    encrypt_file(utils.Key(key), infile, infile + ".aes")
    os.remove(infile)