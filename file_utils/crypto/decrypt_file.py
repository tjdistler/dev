# Decrypts a file using AES GCM (ASE128, 96-bit nonce for NIST SP 800-38D compatibility).
# The key will be prompted for if this script is executed directly. The encrypted file
# will be removed.
#
# Usage: python decrypt_file.py <file_name>

import os
import sys
import json
from base64 import b64decode
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt
from Cryptodome.Random import get_random_bytes
import utils


# Decrypts the ciphertext infile and writes the result to outfile.
# 
# The infile format must be:
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
def decrypt_file(infile, outfile, key=None, key_obj=None):

    with open(infile, "rb") as f:
        contents = f.read()
        # Copy all bytes up to 0x0A0A into a new bytearray
        offset = contents.find(b"\n\n")
        header = bytearray(contents[:offset])
        ciphertext = bytearray(contents[offset + 2:])
    
    # Parse the JSON header
    header = json.loads(header.decode('utf-8'))
    if header["version"] != 1:
        raise ValueError("Unsupported header version: " + str(header["version"]))
    if header["key_params"]["kdf"] != "scrypt":
        raise ValueError("Unsupported key derivation function: " + header["key_params"]["kdf"])

    # Decode values
    dklen = int(header["key_params"]["kdf_params"]["dklen"])
    N = int(header["key_params"]["kdf_params"]["N"])
    r = int(header["key_params"]["kdf_params"]["r"])
    p = int(header["key_params"]["kdf_params"]["p"])
    salt = b64decode(header["key_params"]["kdf_params"]["salt"])
    nonce = b64decode(header["nonce"])
    tag = b64decode(header["tag"])

    if key_obj is None:
        if key is None:
            key = utils.prompt_for_key()
        print(" Setting up key...")
        key_obj = utils.Key(key, salt=salt, dkey_len=dklen, N=N, r=r, p=p)

    # Decrypt the ciphertext
    print(" Decrypting {}...".format(infile))
    cipher = AES.new(key_obj.dkey, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)

    # Write the plaintext to the output file
    with open(outfile, "wb") as f:
        f.write(plaintext)


if __name__ == "__main__":
    infile = sys.argv[1]
    decrypt_file(infile, os.path.splitext(infile)[0])
    os.remove(infile)