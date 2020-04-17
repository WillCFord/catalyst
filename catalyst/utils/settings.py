from typing import Any, Dict, List, Tuple
import configparser
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ("ConfigFileFinder", "MergedConfigParser")


class ConfigFileFinder:
    """Encapsulate the logic for finding and reading config files."""

    def __init__(self, program_name: str) -> None:
        """Initialize object to find config files.

        Args:
            program_name (str): Name of the current program (e.g., catalyst).
        """
        # user configuration file
        self.program_name = program_name
        self.user_config_file = self._user_config_file(program_name)

        # list of filenames to find in the local/project directory
        self.project_filenames = ("setup.cfg", "tox.ini", f".{program_name}")

        self.local_directory = os.path.abspath(os.curdir)

    @staticmethod
    def _user_config_file(program_name: str) -> str:
        if os.name == "nt":  # if running on Windows
            home_dir = os.path.expanduser("~")
            config_file_basename = f".{program_name}"
        else:
            home_dir = os.environ.get(
                "XDG_CONFIG_HOME", os.path.expanduser("~/.config")
            )
            config_file_basename = program_name

        return os.path.join(home_dir, config_file_basename)

    @staticmethod
    def _read_config(
        *files: str,
    ) -> Tuple[configparser.RawConfigParser, List[str]]:
        config = configparser.RawConfigParser()

        found_files: List[str] = []
        for filename in files:
            try:
                found_files.extend(config.read(filename))
            except UnicodeDecodeError:
                logger.exception(
                    f"There was an error decoding a config file."
                    f" The file with a problem was {filename}."
                )
            except configparser.ParsingError:
                logger.exception(
                    f"There was an error trying to parse a config file."
                    f" The file with a problem was {filename}."
                )

        return config, found_files

    def local_config_files(self) -> List[str]:
        """Find all local config files which actually exist.

        Returns:
            List[str]: List of files that exist that are
                local project config  files with extra config files
                appended to that list (which also exist).
        """

        def generate_possible_local_files():
            """Find and generate all local config files."""
            parent = tail = os.getcwd()
            found_config_files = False
            while tail and not found_config_files:
                for project_filename in self.project_filenames:
                    filename = os.path.abspath(
                        os.path.join(parent, project_filename)
                    )
                    if os.path.exists(filename):
                        yield filename
                        found_config_files = True
                        self.local_directory = parent
                (parent, tail) = os.path.split(parent)

        return list(generate_possible_local_files())

    def local_configs(self):
        """Parse all local config files into one config object."""
        config, found_files = self._read_config(*self.local_config_files())
        if found_files:
            logger.debug(f"Found local configuration files: {found_files}")
        return config

    def user_config(self):
        """Parse the user config file into a config object."""
        config, found_files = self._read_config(self.user_config_file)
        if found_files:
            logger.debug(f"Found user configuration files: {found_files}")
        return config


class MergedConfigParser:
    """Encapsulate merging different types of configuration files.

    This parses out the options registered that were specified in the
    configuration files, handles extra configuration files, and returns
    dictionaries with the parsed values.
    """

    #: Set of actions that should use the
    #: :meth:`~configparser.RawConfigParser.getbool` method.
    GETBOOL_ACTIONS = {"store_true", "store_false"}

    def __init__(self, config_finder: ConfigFileFinder):
        """Initialize the MergedConfigParser instance.

        Args:
            config_finder (ConfigFileFinder): Initialized ConfigFileFinder.
        """
        self.program_name = config_finder.program_name
        self.config_finder = config_finder

    def _normalize_value(self, option, value):
        final_value = option.normalize(
            value, self.config_finder.local_directory
        )
        logger.debug(
            f"{value} has been normalized to {final_value}"
            f" for option '{option.config_name}'",
        )
        return final_value

    def _parse_config(self, config_parser):
        config_dict: Dict[str, Any] = {}
        if config_parser.has_section(self.program_name):
            config_dict = {
                option_name: config_parser.get(self.program_name, option_name)
                for option_name in config_parser.options(self.program_name)
            }
        return config_dict

    def parse(self) -> dict:
        """Parse and return the local and user config files.

        First this copies over the parsed local configuration and then
        iterates over the options in the user configuration and sets them if
        they were not set by the local configuration file.

        Returns:
            dict: Dictionary of the parsed and merged configuration options.
        """
        user_config = self._parse_config(self.config_finder.user_config())
        config = self._parse_config(self.config_finder.local_configs())

        for option, value in user_config.items():
            config.setdefault(option, value)

        return config
