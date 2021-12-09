import logging


class CustomError(Exception):
    def __init__(self, code: int, msg: str) -> None:
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return f"[{self.code}]{self.msg}"


class Msg:
    SUCCESS = "SUCCESS"  # 성공
    WRONG_FORMAT = "잘못된 입력 형식입니다."


class Code:
    SUCCESS = 200  # 성공
    BAD_REQUEST = 400


s = "jyp"
try:
    if s == "jyp":
        raise CustomError(Code.BAD_REQUEST, Msg.WRONG_FORMAT)
except CustomError as e:
    logging.info(e)
