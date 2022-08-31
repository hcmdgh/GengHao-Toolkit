from .imports import * 

__all__ = [
    'dump_csv',
    'load_csv',
]


def dump_csv(data: list[dict[str, Any]],
             filepath: str,
             encoding: str = 'utf-8'):
    with open(filepath, 'w', encoding=encoding, newline='') as fp:
        field_names = list(data[0].keys())
        
        writer = csv.DictWriter(fp, fieldnames=field_names)
        writer.writeheader()
        
        for row in data:
            writer.writerow(row)


def load_csv(filepath: str,
             encoding: str = 'utf-8') -> list[dict[str, Any]]:
    with open(filepath, 'r', encoding=encoding, newline='') as fp:
        reader = csv.DictReader(fp)
        
        entry_list = list(reader)
        
        return entry_list
