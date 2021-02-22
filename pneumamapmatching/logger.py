import logging
import pneumapackage.settings
import inspect
import os
import datetime


def log(message, level=None, print_message=False):
    """
    Record a message in the log file or/and print to the console

    Parameters
    ----------
    message : string
    level : int
        logger level
    print_message : print the log message
    Returns
    -------
    None
    """
    if level is None:
        level = pneumapackage.settings.log_level
    if pneumapackage.settings.log_to_file:
        # create a new logger with the calling script's name, or access the existing one
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        logger = get_logger(mod.__name__)
        if level == logging.DEBUG:
            logger.debug(message)
        elif level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
    if print_message:
        print(message)


def get_logger(name):
    """
    Create a logger or return the current one if already instantiated.

    Parameters
    ----------
    level : int
        one of the logger.level constants
    name : string
        name of the logger
    filename : string
        name of the log file

    Returns
    -------
    logger.logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    filename = pneumapackage.settings.log_filename

    # if a logger with this name is not already set up
    if not getattr(logger, 'handler_set', None):

        # get today's date and construct a log filename
        todays_date = datetime.datetime.today().strftime('%Y_%m_%d')
        log_filename = os.path.join(pneumapackage.settings.log_folder, '{}_{}.log'.format(filename, todays_date))

        # if the logs folder does not already exist, create it
        if not os.path.exists(pneumapackage.settings.log_folder):
            os.makedirs(pneumapackage.settings.log_folder)

        # create file handler and log formatter and set them up
        handler = logging.FileHandler(log_filename, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s %(levelname)s @%(name)s.py: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.handler_set = True

    return logger
