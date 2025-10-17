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

@lru_cache
def setup_logger():
    logger = logging.getLogger()
    logger.handlers.clear()
    
    if not PRODUCTION:
        # Development: gunakan Rich untuk pretty logs
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True
        )
        logger.addHandler(handler)
    else:
        # Production: gunakan JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()