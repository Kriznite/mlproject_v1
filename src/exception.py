import sys
import logging

#import traceback

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # get the last value
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message=("error occured in python script name [{0}]"
        "line number [{1}]" 
        "error message[{2}]"
        ).format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

# def extract_traceback(error, tb:sys):
#     # Get exception info from sys
#     exc_typ,exc_val,exc_tb = tb.exc_info()
#     #Extract trace back as a list of FrameSummary objects
#     tb_list = traceback.extract_tb(exc_tb)

#     # collect info from all frames
#     info =[]
#     for frame in tb_list:
#         filename = frame.filename
#         line = frame.lineno
#         code = frame.line
#         info.append[filename,line,code]
#     return info
    
class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message) #call class constructor
        # Add custom attribute
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
