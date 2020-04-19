# flake8: noqa
import logging
import os

from catalyst.utils.tools import settings

from .criterion import CriterionAggregatorCallback
from .cutmix_callback import CutmixCallback
from .knn import KNNMetricCallback
from .optimizer import SaveModelGradsCallback
from .telegram_logger import TelegramLogger

logger = logging.getLogger(__name__)

try:
    import alchemy
    from .alchemy import AlchemyLogger
except (ModuleNotFoundError, ImportError) as ex:
    if settings.alchemy_logger_required:
        logger.exception(
            "alchemy not available, to install alchemy,"
            " run `pip install alchemy`."
        )
        raise ex

try:
    import neptune
    from .neptune import NeptuneLogger
except (ModuleNotFoundError, ImportError) as ex:
    if settings.neptune_logger_required:
        logger.exception(
            "neptune not available, to install neptune,"
            " run `pip install neptune-client`."
        )
        raise ex

try:
    import wandb
    from .wandb import WandbLogger
except (ModuleNotFoundError, ImportError) as ex:
    if settings.wandb_logger_required:
        logger.exception(
            "wandb not available, to install wandb,"
            " run `pip install wandb`."
        )
        raise ex
