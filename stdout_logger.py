from logger import logger


class stdout_logger(logger):


    """
    Constructor
    """
    def __init__(self, log_level):
        logger.__init__(self, log_level)


    """
    log
    Logs the message if log_level is less than or equal to
    the class' threshold.
    """
    def log(self, log_level, message):
        if log_level <= self.__log_level__:
            print(str(log_level) + ": " + message)
        return
