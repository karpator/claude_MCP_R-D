import os


class ApplicationLoggingConfig:
    """
    ApplicationLoggingConfig class is used to set the logging configuration for the application.
    """
    APPLICATION_NAME: str = str(os.getenv("ApplicationName", "pluto-pocketflows-dev"))
    ENVIRONMENT: str = str(os.getenv("Environment", "dev"))
    LOG_LEVEL: str = str(os.getenv("LoggingLogLevel", "DEBUG"))
    # file logging
    IS_FILE_HANDLER_LOG: bool = eval(os.getenv("LoggingIsFileHandlerLog", "False"))
    FILE_HANDLER_LOG_FILE_NAME: str = str(os.getenv("LoggingFileHandlerLogFileName", "spam.log"))
    FILE_HANDLER_LOG_LEVEL: str = str(os.getenv("LoggingFileHandlerLogLevel", "DEBUG"))
    FILE_HANDLER_FORMATTER: str = str(os.getenv("LoggingFileHandlerFormatter", "[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s - %(lineno)d','%Y-%m-%d %H:%M:%S"))
    # console logging
    IS_STREAM_HANDLER_LOG: bool = eval(os.getenv("LoggingIsStreamHandlerLog", "True"))
    STREAM_HANDLER_LOG_LEVEL: str = str(os.getenv("LoggingStreamHandlerLogLevel", "DEBUG"))
    STREAM_HANDLER_FORMATTER: str = str(os.getenv("LoggingFileHandlerFormatter", "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s - %(lineno)d"))
