from .str_util import * 

import hashlib 
import random 
from typing import Any 

__all__ = [
    'hash_by_SHA256',
    'hash_by_SHA1',
    'hash_by_MD5',
    'hash_password',
    'verify_password',
    'hash_obj', 
]


def hash_by_SHA256(b: bytes) -> str:
    s = hashlib.sha256(b).hexdigest()
    assert isinstance(s, str)
    return s 


def hash_by_SHA1(b: bytes) -> str:
    s = hashlib.sha1(b).hexdigest()
    assert isinstance(s, str)
    return s 


def hash_by_MD5(b: bytes) -> str:
    s = hashlib.md5(b).hexdigest()
    assert isinstance(s, str)
    return s 


def hash_password(password: str) -> str:
    salt = generate_random_str(64, lowercase=True, digit=True) 
    salted = password + salt + password + salt 
    
    sha256 = hash_by_SHA256(salted.encode())
    assert isinstance(sha256, str) and len(sha256) == 64
    
    res = salt + sha256 
    
    return res 


def verify_password(password: str,
                    digest: str) -> bool:
    assert len(digest) == 64 + 64  
    salt = digest[:64]
    sha256 = digest[64:]
    salted = password + salt + password + salt 
    
    return hash_by_SHA256(salted.encode()) == sha256 


def hash_obj(obj: Any) -> str:
    raise NotImplementedError
    
    if isinstance(obj, str):
        _bytes = obj.encode(encoding='utf-8')
