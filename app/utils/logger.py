"""
Modul logger untuk aplikasi chatbot menggunakan Rich untuk console output.

Kombinasi antara Rich (untuk console yang indah) dan standard logging (untuk file dan production).

Fitur:
- Rich console output dengan syntax highlighting, pretty printing, dan traceback indah
- JSON format untuk production logging ke file
- File rotation untuk menghindari file log terlalu besar
- Integration dengan FastAPI middleware untuk request logging
- Structured logging dengan metadata tambahan
- Custom themes dan styling

Catatan: Rich digunakan untuk console, JSON untuk file logging.
"""

import logging
import sys
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

# Rich imports
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.columns import Columns
from rich.text import Text
from rich.theme import Theme
from rich import box

# Standard logging imports
from pythonjsonlogger import jsonlogger
from fastapi import Request, Response
import time


# Install Rich traceback untuk menangkap exception dengan lebih indah
install_rich_traceback(show_locals=True, max_frames=100)

# Custom theme untuk aplikasi
custom_theme = Theme({
    "debug": "dim cyan",
    "info": "white",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
    "highlight": "bold magenta",
    "path": "cyan",
    "filename": "bold cyan",
    "line_number": "dim",
    "log.time": "dim green",
    "log.level": "bold",
    "log.message": "white",
    "log.path": "dim yellow",
    "request.method": "bold blue",
    "request.path": "cyan",
    "request.status": "green",
    "request.error": "red",
})

# Global console instance dengan theme custom
console = Console(theme=custom_theme, record=True)


class RichJSONFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter untuk structured logging di production.
    Sama seperti sebelumnya tapi dengan beberapa enhancement.
    """

    def add_fields(self, log_record: Dict[str, Any], record, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)

        # Tambahkan timestamp dalam format ISO
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()

        # Tambahkan informasi aplikasi
        log_record['level'] = record.levelname
        log_record['app'] = 'chatbot_app'
        log_record['environment'] = os.getenv('ENV', 'development')

        # Tambahkan context jika tersedia
        for attr in ['request_id', 'user_id', 'session_id', 'agent_id']:
            if hasattr(record, attr):
                log_record[attr] = getattr(record, attr)


class RichRequestHandler(RichHandler):
    """
    Custom RichHandler yang khusus untuk display request logs.
    """

    def emit(self, record):
        """Override emit untuk format request yang lebih indah."""
        try:
            # Jika ini log request, buat format khusus
            if hasattr(record, 'method') and hasattr(record, 'path'):
                self._emit_request_log(record)
            else:
                # Gunakan emit standar Rich untuk log non-request
                super().emit(record)
        except Exception:
            self.handleError(record)

    def _emit_request_log(self, record):
        """
        Format khusus untuk HTTP request logs menggunakan Rich.
        """
        # Check if this is a complete request log or just debug log
        is_complete_request = hasattr(
            record, 'status_code') and hasattr(record, 'process_time')

        if is_complete_request:
            # This is a completed request log with full info
            # Tentukan warna status code
            status_color = "green"
            if record.status_code >= 400:
                status_color = "red"
            elif record.status_code >= 300:
                status_color = "yellow"

            # Buat text dengan styling
            method_text = Text(record.method, style="bold blue")
            path_text = Text(record.path, style="cyan")
            status_text = Text(f"[{record.status_code}]", style=status_color)
            time_text = Text(f"{record.process_time:.3f}s", style="dim yellow")
            ip_text = Text(
                getattr(record, 'client_ip', 'unknown'), style="dim")

            # Combine text components
            message = Text.assemble(
                method_text, " ",
                path_text, " ",
                status_text, " - ",
                time_text, " - ",
                ip_text, " - ",
                record.getMessage()
            )

            # Log dengan level yang sesuai
            if record.status_code >= 500:
                console.log(message, style="error")
            elif record.status_code >= 400:
                console.log(message, style="warning")
            else:
                console.log(message)
        else:
            # This is a debug log (request started) - simpler format
            method_text = Text(
                getattr(record, 'method', 'UNKNOWN'), style="bold blue")
            path_text = Text(getattr(record, 'path', '/unknown'), style="cyan")
            ip_text = Text(
                getattr(record, 'client_ip', 'unknown'), style="dim")

            # Simple format for debug logs
            message = Text.assemble(
                method_text, " ",
                path_text, " - ",
                ip_text, " - ",
                record.getMessage()
            )

            console.log(message, style="dim")


def setup_rich_logger(
    name: str = 'chatbot_app',
    level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: str = 'logs',
    json_format: bool = True,  # Default JSON untuk file
    show_time: bool = True,
    show_path: bool = True,
    show_level: bool = True,
    markup: bool = True,
    rich_tracebacks: bool = True
) -> logging.Logger:
    """
    Setup logger dengan Rich untuk console dan JSON untuk file.

    Args:
        name: Nama logger
        level: Level log minimum
        log_to_file: Simpan ke file
        log_to_console: Tampilkan di console dengan Rich
        log_dir: Directory file log
        json_format: Format file (JSON recommended untuk production)
        show_time: Tampilkan waktu di console
        show_path: Tampilkan path file di console
        show_level: Tampilkan level di console
        markup: Izinkan Rich markup dalam pesan
        rich_tracebacks: Gunakan Rich untuk exception traceback

    Returns:
        logging.Logger: Logger yang sudah dikonfigurasi
    """

    # Buat logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Hapus handler yang sudah ada
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Rich console handler
    if log_to_console:
        rich_handler = RichHandler(
            console=console,
            show_time=show_time,
            show_path=show_path,
            show_level=show_level,
            markup=markup,
            rich_tracebacks=rich_tracebacks,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
            enable_link_path=True  # Enable clickable paths
        )

        # Set handler level AFTER creating handler
        # Rich handler kadang punya masalah dengan level yang diset di constructor
        # Terima semua level, let logger decide
        rich_handler.setLevel(logging.NOTSET)

        # Force handler untuk terima semua level
        rich_handler.filter = lambda record: True

        # Format sederhana untuk Rich karena Rich menangani styling
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)

    # File handlers dengan JSON format
    if log_to_file:
        # Buat directory log
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = RotatingFileHandler(
            filename=log_path / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)

        # Daily handler
        daily_handler = TimedRotatingFileHandler(
            filename=log_path / f"{name}_daily.log",
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        daily_handler.setLevel(logging.DEBUG)

        # JSON formatter untuk file (production-ready)
        json_formatter = RichJSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )

        file_handler.setFormatter(json_formatter)
        daily_handler.setFormatter(json_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(daily_handler)

    return logger


def get_rich_logger(name: str = None) -> logging.Logger:
    """
    Mendapatkan logger instance yang sudah dikonfigurasi dengan Rich.

    Args:
        name: Nama logger (default: module name)

    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else 'chatbot_app'

    logger = logging.getLogger(name)

    # Get current LOG_LEVEL from environment
    current_log_level = os.getenv('LOG_LEVEL', 'INFO')

    # Setup jika belum dikonfigurasi ATAU jika level berbeda dari yang diharapkan
    expected_level = getattr(logging, current_log_level.upper())
    if not logger.handlers or logger.level != expected_level:
        # Force reconfigure dengan level yang benar
        setup_rich_logger(name, level=current_log_level)

    return logger


async def log_request_rich(request: Request, call_next):
    """
    Middleware untuk logging HTTP request dengan Rich formatting.

    Menghasilkan log yang lebih indah di console dan structured data di file.
    """
    logger = get_rich_logger('chatbot_app.requests')

    # Setup rich handler khusus untuk request
    rich_request_handler = RichRequestHandler(console=console, show_path=False)
    rich_request_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_request_handler)

    # Catat waktu mulai
    start_time = time.time()

    # Ekstrak informasi request
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    query_params = dict(request.query_params)

    # Generate request ID
    request_id = request.headers.get(
        'X-Request-ID', f"req_{int(time.time() * 1000)}")

    # Log request dimulai (hanya di debug level)
    logger.debug(f"[dim]Started {method} {path}[/dim]", extra={
        'method': method,
        'path': path,
        'query_params': query_params,
        'client_ip': client_ip,
        'request_id': request_id
    })

    # Proses request
    response = await call_next(request)

    # Hitung waktu proses
    process_time = time.time() - start_time

    # Log dengan informasi lengkap
    logger.info(
        f"Completed in {process_time:.3f}s",
        extra={
            'method': method,
            'path': path,
            'status_code': response.status_code,
            'process_time': process_time,
            'client_ip': client_ip,
            'request_id': request_id,
            'query_params': query_params,
            'response_headers': dict(response.headers)
        }
    )

    # Tambahkan request ID ke response headers
    response.headers['X-Request-ID'] = request_id

    # Remove handler khusus
    logger.removeHandler(rich_request_handler)

    return response


def log_table(data: Union[Dict, list], title: str = "Data", console_obj: Console = None):
    """
    Helper untuk menampilkan data dalam bentuk tabel menggunakan Rich.

    Args:
        data: Dictionary atau list untuk ditampilkan
        title: Judul tabel
        console_obj: Console object (default: global console)
    """
    console_obj = console_obj or console

    # Buat tabel
    table = Table(title=title, box=box.ROUNDED)

    if isinstance(data, dict):
        # Untuk dictionary, tampilkan key-value
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        for key, value in data.items():
            # Pretty format value
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, indent=2)
                value_syntax = Syntax(
                    value_str, "json", theme="monokai", line_numbers=False)
                table.add_row(str(key), value_syntax)
            else:
                table.add_row(str(key), str(value))

    elif isinstance(data, list) and data:
        # Untuk list of dictionaries
        if isinstance(data[0], dict):
            # Ambil kolom dari item pertama
            columns = list(data[0].keys())
            for col in columns:
                table.add_column(col, style="white")

            # Tambahkan data
            for item in data:
                row = [str(item.get(col, '')) for col in columns]
                table.add_row(*row)
        else:
            # List sederhana
            table.add_column("Item", style="white")
            for item in data:
                table.add_row(str(item))

    console_obj.print(table)


def log_panel(message: str, title: str = "", style: str = "info", console_obj: Console = None):
    """
    Helper untuk menampilkan pesan dalam panel menggunakan Rich.

    Args:
        message: Pesan untuk ditampilkan
        title: Judul panel
        style: Style panel (info, warning, error, success)
        console_obj: Console object
    """
    console_obj = console_obj or console

    # Tentukan warna berdasarkan style
    styles = {
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "success": "green",
        "debug": "dim cyan"
    }

    panel_style = styles.get(style, "white")

    # Buat dan tampilkan panel
    panel = Panel(
        message,
        title=title,
        border_style=panel_style,
        expand=False
    )

    console_obj.print(panel)


def log_code(code: str, language: str = "python", theme: str = "monokai", console_obj: Console = None):
    """
    Helper untuk menampilkan kode dengan syntax highlighting.

    Args:
        code: Kode untuk ditampilkan
        language: Bahasa pemrograman
        theme: Theme syntax highlighting
        console_obj: Console object
    """
    console_obj = console_obj or console

    syntax = Syntax(code, language, theme=theme, line_numbers=True)
    console_obj.print(syntax)


# Context manager untuk logging operasi
class LogOperation:
    """
    Context manager untuk logging operasi dengan Rich.

    Contoh penggunaan:
        with LogOperation("Connecting to database", logger):
            # Operasi yang memakan waktu
            pass
    """

    def __init__(self, operation: str, logger: logging.Logger, level: int = logging.INFO):
        self.operation = operation
        self.logger = logger
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"[dim]Starting:[/dim] {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.log(
                self.level,
                f"[bold green]✓[/bold green] {self.operation} [dim]({duration:.3f}s)[/dim]"
            )
        else:
            self.logger.error(
                f"[bold red]✗[/bold red] {self.operation} failed [dim]({duration:.3f}s)[/dim]",
                exc_info=(exc_type, exc_val, exc_tb)
            )


# Decorator untuk function logging dengan Rich
def log_function_call(logger: logging.Logger = None):
    """
    Decorator untuk logging function calls dengan Rich.

    Contoh:
        @log_function_call()
        def process_data(data):
            # Process data
            return result
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_rich_logger(func.__module__)

        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            with LogOperation(f"Function call: {func_name}", logger, logging.DEBUG):
                # Log arguments di level debug
                if args or kwargs:
                    logger.debug(f"Arguments: args={args}, kwargs={kwargs}")

                result = func(*args, **kwargs)

                # Log result di level debug
                logger.debug(f"Result: {type(result).__name__}")

                return result

        return wrapper
    return decorator


# Default logger dengan Rich
logger = setup_rich_logger(
    name='chatbot_app',
    level=os.getenv('LOG_LEVEL', 'INFO'),
    log_to_file=True,
    log_to_console=True,
    json_format=True,  # JSON untuk file, Rich untuk console
    show_time=True,
    show_path=True,
    rich_tracebacks=True,
    markup=True
)

# Tambahkan console save untuk debugging
if os.getenv('SAVE_CONSOLE', 'false').lower() == 'true':
    console.save_html("logs/console.html")

# Export yang diperlukan
__all__ = [
    'setup_rich_logger',
    'get_rich_logger',
    'log_request_rich',
    'log_table',
    'log_panel',
    'log_code',
    'LogOperation',
    'log_function_call',
    'logger',
    'console',
    'custom_theme'
]
