import sys
import traceback
import logging


def error_message_detail(error, error_detail=None):
    """
    Returns detailed error message with filename and line number.
    """
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script [{file_name}] at line [{line_number}]: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail=None):
        super().__init__(error) 
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message


'''if __name__=='__main__':
    try:
        a = 1/0
    except:
        logging.info("divide by zero")
        raise CustomException    '''        