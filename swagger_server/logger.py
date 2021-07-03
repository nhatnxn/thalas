import logging


def create_logger(name, level='info', logfile=None):
    global ADD_HANDLE
    logger = logging.getLogger(name)

    if level == 'info':
        lvl = logging.INFO
    elif level == 'warning':
        lvl = logging.WARNING
    else:
        lvl = logging.DEBUG
    logger.setLevel(lvl)
    # create console handler and set level to debug
    if logfile is None:
        ch = logging.StreamHandler()
        ch.setLevel(lvl)
    else:
        ch = logging.FileHandler(logfile)
        ch.setLevel(lvl)

    # create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger