# {
#     code:'',
#     msg:'',
#     data:'',
#     request_time:'',
#     response_time:''
# }
from datetime import datetime

class ApiResponse():

    @classmethod
    def success(cls, status: int, msg: str, data: object, request_time: datetime) -> dict:
        return {
            'status': status,
            'msg': msg,
            'data': data,
            'request_time': request_time,
            'response_time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
        }

    @classmethod
    def fail(cls, status: int, msg: str, request_time: datetime) -> dict:
        return {
            'status': status,
            'msg': msg,
            'data': None,
            'request_time': request_time,
            'response_time': datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
        }

# enum class for error code, msg
class Msg:
    SUCCESS = 'SUCCESS' # 성공
    WRONG_FORMAT = '잘못된 입력 형식입니다.'

class Status:
    SUCCESS = 200 # 성공
    BAD_REQUEST = 400