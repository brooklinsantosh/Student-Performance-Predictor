import sys
from src.logger import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Builds custom errror message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    error_message= f"Error occured in python script name [{file_name}] line number [{line_num}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys) -> None:
        super().__init__(error)
        self.error_message= error_message_detail(error, error_detail)
        logging.error(self.error_message)
    
    def __str__(self) -> str:
        return self.error_message