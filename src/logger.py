import logging 
import os
from datetime import datetime

#Create the logs directory if it doesn't exist
logs_dir=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_dir,exist_ok=True)

#Define the log file name and path
LOG_FILE_NAME=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH= os.path.join(logs_dir,LOG_FILE_NAME)

#configure loggng
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    )

logger = logging.getLogger(__name__)

