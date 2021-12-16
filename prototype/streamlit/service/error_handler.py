import logging

class CustomError(Exception):
    def __init__(self, code: int, msg: str) -> None:
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return f'[{self.code}]{self.msg}'
    
