import logging
import pickle

from catalyst.utils.tools import settings

logger = logging.getLogger(__name__)

try:
    import pyarrow
except (ModuleNotFoundError, ImportError) as ex:
    if settings.use_pyarrow:
        logger.exception(
            "pyarrow not available, switching to pickle."
            " to install pyarrow, run `pip install pyarrow`."
        )
        raise ex


def pyarrow_serialize(data):
    """Serialize the data into bytes using pyarrow.

    Args:
        data: a value

    Returns:
        Returns a bytes object serialized with pyarrow data.
    """
    return pyarrow.serialize(data).to_buffer().to_pybytes()


def pyarrow_deserialize(data):
    """Deserialize bytes into an object using pyarrow.

    Args:
        bytes: a bytes object containing serialized with pyarrow data.

    Returns:
        Returns a value deserialized from the bytes-like object.
    """
    return pyarrow.deserialize(data)


def pickle_serialize(data):
    """Serialize the data into bytes using pickle.

    Args:
        data: a value

    Returns:
        Returns a bytes object serialized with pickle data.
    """
    return pickle.dumps(data)


def pickle_deserialize(data):
    """Deserialize bytes into an object using pickle.

    Args:
        bytes: a bytes object containing serialized with pickle data.

    Returns:
        Returns a value deserialized from the bytes-like object.
    """
    return pickle.loads(data)


if settings.use_pyarrow:
    serialize = pyarrow_serialize
    deserialize = pyarrow_deserialize
else:
    serialize = pickle_serialize
    deserialize = pickle_deserialize

__all__ = ["serialize", "deserialize"]
