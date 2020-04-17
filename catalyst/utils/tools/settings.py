from catalyst.utils.settings import ConfigFileFinder, MergedConfigParser

_SETTINGS = MergedConfigParser(ConfigFileFinder("catalyst")).parse()

# [catalyst]
USE_CONTRIB = _SETTINGS.get("load_catalyst_contrib", False)
USE_CV = _SETTINGS.get("load_catalyst_cv", False)
USE_ML = _SETTINGS.get("load_catalyst_ml", False)
USE_NLP = _SETTINGS.get("load_catalyst_nlp", False)

STATE_MAIN_METRIC = _SETTINGS.get("state_main_metric", "loss")

# stages
STAGE_TRAIN_PREFIX = _SETTINGS.get("stage_train_prefix", "train")
STAGE_VALID_PREFIX = _SETTINGS.get("stage_valid_prefix", "valid")
STAGE_INFER_PREFIX = _SETTINGS.get("stage_infer_prefix", "infer")

# loader
LOADER_TRAIN_PREFIX = _SETTINGS.get("loader_train_prefix", "train")
LOADER_VALID_PREFIX = _SETTINGS.get("loader_valid_prefix", "valid")
LOADER_INFER_PREFIX = _SETTINGS.get("loader_infer_prefix", "infer")

# callbacks
CHECK_RUN_NUM_BATCH_STEPS = _SETTINGS.get("check_run_num_batch_steps", 2)
CHECK_RUN_NUM_EPOCH_STEPS = _SETTINGS.get("check_run_num_epoch_steps", 2)

# other
# @TODO: check if catalyst env variables
# USE_DDP = _SETTINGS.get("use_ddp", False)
# USE_APEX = _SETTINGS.get("use_apex", False)

# [catalyst-contrib]
USE_ALCHEMY = _SETTINGS.get("load_alchemy_logger", USE_CONTRIB)
USE_NEPTUNE = _SETTINGS.get("load_neptune_logger", USE_CONTRIB)
USE_WANDB = _SETTINGS.get("load_wandb_logger", USE_CONTRIB)
CATALYST_TELEGRAM_TOKEN = _SETTINGS.get("telegram_logger_token", None)
CATALYST_TELEGRAM_CHAT_ID = _SETTINGS.get("telegram_logger_chat_id", None)
USE_LZ4 = _SETTINGS.get("load_lz4", USE_CONTRIB)
USE_PYARROW = _SETTINGS.get("load_pyarrow", USE_CONTRIB)

# [catalyst-cv]
USE_ALBUMENTATIONS = _SETTINGS.get("load_albumentations", USE_CV)
USE_SEGMENTATION_MODELS = _SETTINGS.get("load_segmentation_models", USE_CV)
USE_LIBJPEG_TURBO = _SETTINGS.get("load_libjpeg_turbo", False)

# [catalyst-ml]
USE_NMSLIB = _SETTINGS.get("load_nmslib", USE_ML)

# [catalyst-nlp]
USE_TRANSFORMERS = _SETTINGS.get("load_transformers", USE_NLP)
