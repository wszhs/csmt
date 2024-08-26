
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES
ROOT_PATH='/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox'
CSMT_NUMPY_DTYPE = np.float32
CSMT_DATA_PATH: str

# --------------------------------------------------------------------------------------------- DEFAULT PACKAGE CONFIGS

_folder = os.path.expanduser("~")
if not os.access(_folder, os.W_OK):
    _folder = "/tmp"
_folder = os.path.join(_folder, ".csmt")

def set_data_path(path):
    expanded_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(expanded_path, exist_ok=True)
    logger.info(f"set CSMT_DATA_PATH to %s", expanded_path)
    if not os.access(expanded_path, os.R_OK):
        raise OSError(f"path {expanded_path} cannot be read from")
    if not os.access(expanded_path, os.W_OK):
        logger.warning(f"path %s is read only", expanded_path)
    global CSMT_DATA_PATH
    CSMT_DATA_PATH = expanded_path
    logger.info(f"set CSMT_DATA_PATH to %s", expanded_path)

# Load data from configuration file if it exists. Otherwise create one.
_config_path = os.path.expanduser(os.path.join(_folder, "config.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)

            # Since renaming this variable we must update existing config files
            if "DATA_PATH" in _config:
                _config["CSMT_DATA_PATH"] = _config.pop("DATA_PATH")
                try:
                    with open(_config_path, "w") as f:
                        f.write(json.dumps(_config, indent=4))
                except IOError:
                    logger.warning("Unable to update configuration file", exc_info=True)

    except ValueError:
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:
        logger.warning("Unable to create folder for configuration file.", exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {"CSMT_DATA_PATH": os.path.join(_folder, "data")}
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        logger.warning("Unable to create configuration file", exc_info=True)

if "CSMT_DATA_PATH" in _config:
    set_data_path(_config["CSMT_DATA_PATH"])