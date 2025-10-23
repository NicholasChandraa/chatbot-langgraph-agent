# logger.py
import logging
import json
import sys
import os
from logging import Formatter
from functools import lru_cache
from dotenv import load_dotenv
from rich.logging import RichHandler



load_dotenv()

PRODUCTION = True if os.getenv("ENVIRONMENT") == "production" else False 

class JsonFormatter(Formatter):
    def format(self, record):
        json_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'req'):
            json_record["req"] = record.req
        if hasattr(record, 'res'):
            json_record["res"] = record.res
            
        return json.dumps(json_record)

class ClickablePathFormatter(Formatter):
    """
    Custom formatter that outputs clickable file paths for IDEs
    Format: LEVEL - message (file:line)
    """
    def format(self, record):
        # Get full file path
        filepath = os.path.abspath(record.pathname)
        
        # Format: LEVEL - message (filepath:lineno)
        log_message = f"{record.levelname:8} {record.getMessage()}"
        location = f"{filepath}:{record.lineno}"
        
        # Add function name if available
        if record.funcName and record.funcName != '<module>':
            location = f"{filepath}:{record.funcName}:{record.lineno}"
        
        return f"{log_message:100} {location}"


@lru_cache
def setup_logger():
    logger = logging.getLogger()
    logger.handlers.clear()
    
    if not PRODUCTION:
        # Development: gunakan Rich untuk pretty logs dengan clickable paths
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True,
            enable_link_path=True,  # Enable clickable paths in Rich
            log_time_format="[%X]"
        )
        logger.addHandler(handler)
    else:
        # Production: gunakan JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    
    # Set level based on environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    return logger

logger = setup_logger()