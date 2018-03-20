from logger import logger
import os


class file_logger(logger):
    """
    Constructor
    """
    def __init__(self, log_level, filename="file_log.txt"):
        logger.__init__(self, log_level)
        try:
            self.file = open(filename, "w+")
        except Exception:
            print(Exception)
            exit()

    """
    log
    Logs the message if log_level is less than or equal to
    the class' threshold.
    """
    def log(self, log_level, message):
        if log_level <= self.__log_level__:
            self.file.write(str(log_level) + ": " + message + "\n")
        return
