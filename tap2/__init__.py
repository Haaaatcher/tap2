import sys
from .model import MoE2, AAModel
from .core import TAInfer, TAInput, TAOutput, TABatchInput, TABatchOutput, AAInfer, AAInput, AAOutput
from loguru import logger
from pathlib import Path



__version__ = "0.1.0"

__author__ = "Hang Luo"

__email__ = "haaaatcher@gmail.com"

__all__ = [
    "MoE2",
    "AAModel",
    "TAInfer",
    "TAInput",
    "TAOutput",
    "TABatchInput",
    "TABatchOutput",
    "AAInput",
    "AAOutput",
    "AAInfer"
]

_log_format_ = "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> " \
               "<level>[{level}]</level> " \
               "<cyan>[{name}:{function}:{line}]</cyan> " \
               "<level>{message}</level>"

logger.remove()

logger.add(sys.stderr, format=_log_format_)

logger.add(
    Path.home() / "logs" / "tap2.log",
    level="INFO",
    format=_log_format_,
    rotation="500 MB",
    retention=10,
    compression="zip",
    encoding="utf-8"
)
