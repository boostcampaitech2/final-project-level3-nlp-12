# {
#     code:'',
#     msg:'',
#     data:'',
#     request_time:'',
#     response_time:''
# }
from datetime import datetime
from pprint import pprint
from error_handler import Code, Msg


class ApiResponse:
    @classmethod
    def success(cls, code: int, msg: str, data: object, request_time: datetime) -> dict:
        return {
            "code": code,
            "msg": msg,
            "data": data,
            "request_time": request_time,
            "response_time": datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        }

    @classmethod
    def fail(cls, code: int, msg: str, request_time: datetime) -> dict:
        return {
            "code": code,
            "msg": msg,
            "data": None,
            "request_time": request_time,
            "response_time": datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        }


class TestClass:
    def __init__(self) -> None:
        pass


test_class = TestClass()
pprint(
    ApiResponse.success(
        Code.SUCCESS,
        Msg.SUCCESS,
        test_class,
        datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
    )
)
