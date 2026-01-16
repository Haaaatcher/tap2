import sys
from .model import MoE2
from .core import TAP2Infer, TAP2Exception, TAP2Input, TAP2Output, TAP2BatchInput, TAP2BatchOutput
from loguru import logger
from pathlib import Path



__version__ = "0.1.0"

__author__ = "Hang Luo"

__email__ = "haaaatcher@gmail.com"

__all__ = [
    "MoE2",
    "TAP2Infer",
    "TAP2Exception",
    "TAP2Input",
    "TAP2Output",
    "TAP2BatchInput",
    "TAP2BatchOutput"
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
