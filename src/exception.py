import sys
import traceback


# def error_message_detail(error, error_detail:sys):
#     """
#     Generates a detailed error message string that includes the name of the 
#     python script where the error occurs, the line number, and the error
#     message
#     """
    
#     # extract traceback object from the current exception
#     _,_,exc_tb=error_detail.exc_info()
#     file_name=exc_tb.tb_frame.f_code.co_filename
#     error_message=("error occured in python script name [{0}]"
#         "line number [{1}]" 
#         "error message[{2}]"
#         ).format(
#         file_name,exc_tb.tb_lineno,str(error)
#     )
#     return error_message

def error_message_detail(error:Exception, error_detail:sys) ->str:
    # Extract traceback object from current exception
    tb = traceback.TracebackException.from_exception(error)
    file_name=tb.stack[-1].filename
    line_number=tb.stack[-1].lineno
    error_message=(
        f"Error occured in python script named [{file_name}]"
        f"line number[{line_number}]"
        f"error message[{error}]"
    )

    return error_message
    
class CustomException(Exception):

    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message) 
        # Add custom attribute
        self.error_message=error_message_detail(
            error_message,error_detail=error_detail
            )

    def __str__(self):
        return self.error_message
    
