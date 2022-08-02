from .imports import * 

__all__ = ['MySQLConnection']


class MySQLConnection:
    def __init__(self,
                 *, 
                 host: str,
                 port: int = 3307, 
                 user: str,
                 password: str,
                 database: str,
                 charset: str = 'utf8mb4'):
        self.conn = pymysql.connect(
            host = host,
            port = port, 
            user = user,
            password = password,
            database = database,
            charset = charset,
            autocommit = True, 
            cursorclass = pymysql.cursors.DictCursor,
        )
        
        self.cursor = self.conn.cursor() 

    # def __del__(self):
    #     if hasattr(self, 'cursor'):
    #         self.cursor.__exit__() 
        
    #     if hasattr(self, 'conn'):
    #         self.conn.__exit__()
        
    def scan_table(self,
                   table_name: str,
                   primary_key: str = 'id', 
                   batch_size: int = 1_0000) -> Iterator[dict[str, Any]]:
        last_id = None 
        
        while True:
            if last_id is None:
                self.cursor.execute(
                    f"SELECT * FROM {table_name} ORDER BY {primary_key} ASC LIMIT %s",
                    [batch_size],
                )
            else:
                self.cursor.execute(
                    f"SELECT * FROM {table_name} WHERE {primary_key} > %s ORDER BY {primary_key} ASC LIMIT %s",
                    [last_id, batch_size],
                )
            
            last_id = None 
            
            for entry in self.cursor.fetchall():
                last_id = entry[primary_key]
                yield entry 

            if last_id is None:
                break 
