import datetime
import json
import logging
import sys
import traceback

class JSONFormatter(logging.Formatter):
    """Custom standard logging Formatter to output log records as single-line JSON."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Include tracebacks for errors
        if record.exc_info:
            log_data["exception"] = "".join(traceback.format_exception(*record.exc_info))
            
        # Extract custom 'extra' fields passed via extra={}
        standard_attrs = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
            'relativeCreated', 'stack_info', 'thread', 'threadName'
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                log_data[key] = value
                
        return json.dumps(log_data)


class DynamicStdoutStreamHandler(logging.StreamHandler):
    """A logging StreamHandler that dynamically resolves sys.stdout at write time to support testing capturing."""
    def __init__(self):
        super().__init__()

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def setup_structured_logging():
    """Sets up standard library root logger and overrides Uvicorn access/error/default log handlers to use structured JSON."""
    handler = DynamicStdoutStreamHandler()
    handler.setFormatter(JSONFormatter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
    
    # List of Uvicorn loggers to configure
    loggers_to_override = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi"
    ]
    for logger_name in loggers_to_override:
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.propagate = False
