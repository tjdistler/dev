
import os
from Cryptodome.Protocol.KDF import scrypt
from Cryptodome.Random import get_random_bytes

SCRYPT_SALT_LEN = 16
SCRYPT_DERIVED_KEY_LEN = 16
SCRYPT_N = 2**20
SCRYPT_R = 8
SCRYPT_P = 1

class Key:
    """
    A class to represent an encryption key.
    """

    def __init__(self, key, salt=None, salt_len=SCRYPT_SALT_LEN, dkey_len=SCRYPT_DERIVED_KEY_LEN, N=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P):
        self.key = key
        self.salt_len = salt_len
        if salt is None:
            self.salt = get_random_bytes(salt_len)
        else:
            self.salt = salt
            self.salt_len = len(salt)
        self.dkey_len = dkey_len
        self.N = N
        self.r = r
        self.p = p
        self.dkey = scrypt(key, salt=self.salt, N=self.N, r=self.r, p=self.p, key_len=self.dkey_len)

    def __str__(self):
        return "[salt={}, salt_len={}, dkey_len={}, N={}, r={}, p={}]".format(self.salt, self.salt_len, self.dkey_len, self.N, self.r, self.p)


# Prompt the user to enter their key, hiding the console input.
def prompt_for_key():
    # Hide console input while typing password.
    os.system("stty -echo")
    key = input("Key: ")
    os.system("stty echo")
    print()
    return key.encode('utf-8')