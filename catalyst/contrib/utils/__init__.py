# flake8: noqa
# isort:skip_file

import logging
import os

logger = logging.getLogger(__name__)

from catalyst.utils.tools import settings
from .argparse import boolean_flag
from .compression import pack, pack_if_needed, unpack, unpack_if_needed
from .confusion_matrix import (
    calculate_tp_fp_fn,
    calculate_confusion_matrix_from_arrays,
    calculate_confusion_matrix_from_tensors,
)
from .dataset import create_dataset, split_dataset_train_test, create_dataframe
from .misc import (
    args_are_not_none,
    make_tuple,
    pairwise,
)
from .pandas import (
    dataframe_to_list,
    folds_to_list,
    split_dataframe_train_test,
    split_dataframe_on_folds,
    split_dataframe_on_stratified_folds,
    split_dataframe_on_column_folds,
    map_dataframe,
    separate_tags,
    get_dataset_labeling,
    split_dataframe,
    merge_multiple_fold_csv,
    read_multiple_dataframes,
    read_csv_data,
    balance_classes,
)
from .parallel import parallel_imap, tqdm_parallel_imap, get_pool
from .plotly import plot_tensorboard_log
from .serialization import deserialize, serialize

try:
    from .image import (
        has_image_extension,
        imread,
        imwrite,
        imsave,
        mask_to_overlay_image,
        mimread,
        mimwrite_with_meta,
        tensor_from_rgb_image,
        tensor_to_ndimage,
    )
except (ModuleNotFoundError, ImportError) as ex:
    if settings.cv_required:
        logger.exception(
            "some of catalyst-cv dependencies not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex

try:
    import transformers  # noqa: F401
    from .text import tokenize_text, process_bert_output
except (ModuleNotFoundError, ImportError) as ex:
    if settings.nlp_required:
        logger.exception(
            "some of catalyst-nlp dependencies not available,"
            " to install dependencies, run `pip install catalyst[nlp]`."
        )
        raise ex

from .visualization import (
    plot_confusion_matrix,
    render_figure_to_tensor,
    plot_metrics,
)
