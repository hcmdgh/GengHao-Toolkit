from .imports import * 

__all__ = [
    'json_dump',
    'json_load',
]


def json_dump(data: Any,
              output_file: Optional[str] = None) -> bytes:
    json_bytes = orjson.dumps(data)
    
    if output_file:
        with open(output_file, 'wb') as fp:
            fp.write(json_bytes)
    
    return json_bytes


def json_load(data: Union[bytes, str]) -> Any:
    if isinstance(data, bytes):
        pass 
    elif isinstance(data, str):  # 从文件中读取字节流
        with open(data, 'rb') as fp:
            data = fp.read() 
    else:
        raise TypeError 
    
    return orjson.loads(data)
        