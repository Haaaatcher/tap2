import sys
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from .server import *
from loguru import logger



__version__ = '0.1.1'

__author__ = 'Hang Luo'

__email__ = 'haaaatcher@gmail.com'

__all__ = [
    'TAPhysBaseModel',
    'TAMechBaseModel',
    'TAWFBaseModel',
    'TADSLitAPI',
    'TATCLitAPI',
    'TAECLitAPI',
    'TAYMLitAPI',
    'TABMLitAPI',
    'TASMLitAPI',
    'TAPRLitAPI',
    'TASELitAPI',
    'TASHCLitAPI',
    'TATELitAPI',
    'TABTTLitAPI',
    'TAYSLitAPI',
    'TATSLitAPI',
    'TAHDLitAPI',
    'TAHPLitAPI',
    'TAWFLitAPI',
    'AAPhysBaseModel',
    'AAMechBaseModel',
    'AATCLitAPI',
    'AATELitAPI',
    'AAYSLitAPI',
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

fm.fontManager.addfont(str(resources.files('tap2.resource').joinpath('SimHei.ttf')))

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False
