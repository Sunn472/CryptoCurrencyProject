import sys

def error_message_detail(error,error_detail: sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message=f"Error occured in scripts: [{file_name}] at line number: [{exc_tb.tb_lineno}] with message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self,error,error_detail: sys):
        super().__init__(error)
        self.error_message=error_message_detail(error,error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return CustomException.__name__.str() + f"({self.error_message})"
