import logging

from vllm.logger import (
    DEFAULT_LOGGING_CONFIG,
    init_logger,  # noqa: F401
)

DEFAULT_LOGGER_NAME = __name__.split(".")[0]

config = {**DEFAULT_LOGGING_CONFIG}

config["formatters"][DEFAULT_LOGGER_NAME] = DEFAULT_LOGGING_CONFIG["formatters"]["vllm"]

handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
handler_config["formatter"] = DEFAULT_LOGGER_NAME
config["handlers"][DEFAULT_LOGGER_NAME] = handler_config

logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
logger_config["handlers"] = [DEFAULT_LOGGER_NAME]
config["loggers"][DEFAULT_LOGGER_NAME] = logger_config

logging.config.dictConfig(config)
