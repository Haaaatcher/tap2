import sys
from .model import MoE2
from .core import TAPPInfer, TAPPException, TAPPInput, TAPPOutput, TAPPBatchInput, TAPPBatchOutput
from loguru import logger
from pathlib import Path



__version__ = "0.0.8"

__author__ = "Hang Luo"

__email__ = "haaaatcher@gmail.com"

__all__ = [
    "MoE2",
    "TAPPInfer",
    "TAPPException",
    "TAPPInput",
    "TAPPOutput",
    "TAPPBatchInput",
    "TAPPBatchOutput"
]

_log_format = "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> " \
              "<level>[{level}]</level> " \
              "<cyan>[{name}:{function}:{line}]</cyan> " \
              "<level>{message}</level>"

logger.remove()

logger.add(sys.stderr, format=_log_format)

logger.add(
    Path.home() / "logs" / "tap2.log",
    level="INFO",
    format=_log_format,
    rotation="500 MB",
    retention=10,
    compression="zip",
    encoding="utf-8"
)
