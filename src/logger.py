import os
from datetime import datetime
import logging

LOG_FILE = f"{datetime.now().strftime('%Y-%M-%d_%H-%M-%S')}"

log_path=os.path.join(os.getcwd(),'Logs','Log_File')

os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)