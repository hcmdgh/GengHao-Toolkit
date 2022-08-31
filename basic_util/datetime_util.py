from .imports import * 

__all__ = [
    'date2str',
    'datetime2str',
    'str2date',
    'str2datetime',
]


def date2str(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def datetime2str(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def str2date(s: str) -> date:
    return datetime.strptime(s, '%Y-%m-%d').date()


def str2datetime(s: str) -> datetime:
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
