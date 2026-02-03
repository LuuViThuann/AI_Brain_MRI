"""
logger.py
Centralized logging utility for NeuroScan AI.

Features:
  - Color-coded console output (INFO=cyan, WARN=orange, ERROR=red, SUCCESS=green)
  - Simultaneous file logging (plain text)
  - Per-module named loggers
  - Configurable log levels
  - Timestamp + module name in every log line
  - Simple setup — just import and call

Usage:
    from utils.logger import get_logger

    log = get_logger("MyModule")
    log.info("Server started on port 8000")
    log.warn("Model file not found, using mock predictions")
    log.error("Failed to connect to Groq API")
    log.success("Training complete — Dice: 0.89")
    log.debug("Tensor shape: (1, 256, 256, 1)")  # only shown if level=DEBUG
"""

import logging
import os
import sys
from datetime import datetime


# ============================================================
# COLOR CODES (ANSI escape sequences)
# ============================================================
class _Colors:
    RESET   = '\033[0m'
    CYAN    = '\033[36m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    RED     = '\033[31m'
    ORANGE  = '\033[38;5;208m'
    BOLD    = '\033[1m'
    DIM     = '\033[2m'
    MAGENTA = '\033[35m'
    BLUE    = '\033[34m'

C = _Colors()


# ============================================================
# CUSTOM LOG LEVELS
# ============================================================
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


# ============================================================
# COLORED FORMATTER (for console)
# ============================================================
class _ColoredFormatter(logging.Formatter):
    """Applies ANSI colors based on log level."""

    LEVEL_COLORS = {
        'DEBUG':   C.DIM + C.CYAN,
        'INFO':    C.CYAN,
        'SUCCESS': C.GREEN + C.BOLD,
        'WARNING': C.ORANGE,
        'ERROR':   C.RED + C.BOLD,
        'CRITICAL':C.RED + C.BOLD,
    }

    LEVEL_ICONS = {
        'DEBUG':   '⬡',
        'INFO':    '●',
        'SUCCESS': '✓',
        'WARNING': '⚠',
        'ERROR':   '✗',
        'CRITICAL':'✗✗',
    }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color     = self.LEVEL_COLORS.get(levelname, C.RESET)
        icon      = self.LEVEL_ICONS.get(levelname, '•')

        # Time formatting
        ct = datetime.fromtimestamp(record.created)
        time_str = ct.strftime('%H:%M:%S')

        # Module name (truncated to 16 chars)
        module = record.name[-16:] if len(record.name) > 16 else record.name
        module_padded = module.ljust(16)

        # Build colored line
        line = (
            f"{C.DIM}{time_str}{C.RESET} "
            f"{color}[{icon}]{C.RESET} "
            f"{C.DIM}{module_padded}{C.RESET} "
            f"{color}{record.getMessage()}{C.RESET}"
        )

        # Append exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            import traceback
            exc_text = ''.join(traceback.format_exception(*record.exc_info))
            line += f"\n{C.RED}{exc_text}{C.RESET}"

        return line


# ============================================================
# PLAIN FORMATTER (for file output — no colors)
# ============================================================
class _PlainFormatter(logging.Formatter):
    """Clean format for log files."""

    LEVEL_ICONS = {
        'DEBUG':   '[DBG]',
        'INFO':    '[INF]',
        'SUCCESS': '[OK ]',
        'WARNING': '[WRN]',
        'ERROR':   '[ERR]',
        'CRITICAL':'[CRT]',
    }

    def format(self, record: logging.LogRecord) -> str:
        ct = datetime.fromtimestamp(record.created)
        time_str = ct.strftime('%Y-%m-%d %H:%M:%S')
        icon     = self.LEVEL_ICONS.get(record.levelname, '[   ]')
        module   = record.name[-20:]

        line = f"{time_str} {icon} [{module:>20s}] {record.getMessage()}"

        if record.exc_info and record.exc_info[0] is not None:
            import traceback
            line += '\n' + ''.join(traceback.format_exception(*record.exc_info))

        return line


# ============================================================
# LOGGER REGISTRY — singleton per module name
# ============================================================
_loggers = {}

# Default config
_DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FILE          = os.getenv("LOG_FILE", "logs/neuroscan.log")


def _ensure_log_dir():
    """Create logs/ directory if it doesn't exist."""
    log_dir = os.path.dirname(_LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def get_logger(name: str = "App", level: str = None) -> logging.Logger:
    """
    Get (or create) a named logger with colored console + file output.

    Args:
        name:  Module/component name shown in log output
        level: Override log level for this logger ('DEBUG','INFO','WARNING','ERROR')

    Returns:
        logging.Logger with .info(), .warn(), .error(), .debug(), .success()
    """
    if name in _loggers:
        return _loggers[name]

    # Resolve log level
    level_str = (level or _DEFAULT_LOG_LEVEL).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    # Create logger
    logger = logging.Logger(name, level=numeric_level)

    # --- Console handler (colored) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_ColoredFormatter())
    logger.addHandler(console_handler)

    # --- File handler (plain text) ---
    try:
        _ensure_log_dir()
        file_handler = logging.FileHandler(_LOG_FILE, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(_PlainFormatter())
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        # If file logging fails, continue with console only
        logger.warning(f"Could not set up file logging: {e}")

    # --- Add .success() method ---
    def success(msg, *args, **kwargs):
        logger.log(SUCCESS_LEVEL, msg, *args, **kwargs)

    logger.success = success

    # Cache logger
    _loggers[name] = logger
    return logger


# ============================================================
# CONVENIENCE: Module-level default logger
# ============================================================
def info(msg):    get_logger("App").info(msg)
def warn(msg):    get_logger("App").warning(msg)
def error(msg):   get_logger("App").error(msg)
def debug(msg):   get_logger("App").debug(msg)
def success(msg): get_logger("App").success(msg)


# ============================================================
# DEMO (run standalone)
# ============================================================
if __name__ == "__main__":
    print("=" * 58)
    print("  NeuroScan AI — Logger Demo")
    print("=" * 58)
    print()

    # --- Demo 1: Basic usage ---
    log = get_logger("MainApp")
    log.info("Application starting up...")
    log.success("Connected to database successfully")
    log.warning("Config file not found, using defaults")
    log.error("Failed to load model weights")
    log.debug("This only shows if LOG_LEVEL=DEBUG")

    print()

    # --- Demo 2: Different modules ---
    model_log   = get_logger("ModelEngine")
    groq_log    = get_logger("GroqClient")
    brain3d_log = get_logger("Brain3DRenderer")

    model_log.info("Loading U-Net model from saved_model/")
    model_log.success("Model loaded — 2.4M parameters")

    groq_log.info("Sending prompt to llama3-70b-8192...")
    groq_log.success("Report generated in 0.82s")

    brain3d_log.info("Generating brain mesh (resolution=40)")
    brain3d_log.success("Mesh ready — 6,400 vertices, 12,800 faces")

    print()

    # --- Demo 3: Debug level logger ---
    debug_log = get_logger("DebugTest", level="DEBUG")
    debug_log.debug("Tensor shape: torch.Size([1, 256, 256, 1])")
    debug_log.debug("Prediction confidence: 0.8734")

    print()
    print(f"[✓] Log file saved to: {_LOG_FILE}")
    print("    (Check the file for plain-text version of all logs above)")