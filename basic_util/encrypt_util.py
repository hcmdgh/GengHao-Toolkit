from .imports import * 
from .hash_util import * 

from Crypto.Cipher import AES


__all__ = [
    'encrypt_bytes', 
    'decrypt_bytes', 
]


def _zfill(data: bytes,
           length: int = 16) -> bytes:
    if len(data) > length:
        raise ValueError 
    elif len(data) == length:
        return data 
    else:
        zeros = bytes(length - len(data))
        res = data + zeros 
        return res 


def encrypt_bytes(data: bytes,
                  key: Union[str, bytes]) -> bytes:
    if isinstance(key, str):
        key = key.encode()
        
    key = _zfill(key)
    sha1 = hash_sha1(data).encode() 
    assert len(sha1) == 40
    
    data_with_hash = data + sha1
    
    cipher = AES.new(key, AES.MODE_EAX)
    encrypted = cipher.encrypt(data_with_hash)
    nonce = cipher.nonce     
    assert isinstance(nonce, bytes) and len(nonce) == 16 
    
    res = nonce + encrypted 
    
    return res 


def decrypt_bytes(data: bytes,
                  key: Union[str, bytes]) -> Optional[bytes]:
    nonce = data[:16]
    encrypted = data[16:]
    assert len(nonce) == 16 and len(encrypted) > 0 
                  
    if isinstance(key, str):
        key = key.encode()
        
    key = _zfill(key)
    
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    decrypted = cipher.decrypt(encrypted)

    raw_data = decrypted[:-40]
    sha1 = decrypted[-40:]
    
    if hash_sha1(raw_data).encode() != sha1:
        return None     
    else:
        return raw_data 
