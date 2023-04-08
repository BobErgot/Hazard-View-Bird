import os
from datetime import datetime

def log_to_file(log_dir, message):
    """
    Appends a message to a log file within the specified directory.
    Creates the directory and file if they do not exist.

    Args:
        log_dir (str): The directory where the log file will be stored.
        message (str): The message to log.
    """
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    log_file_path = os.path.join(log_dir, "training_log.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")
