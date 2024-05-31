import logging

from vllm.logger import (
    DEFAULT_LOGGING_CONFIG,
    init_logger,  # noqa: F401
)

config = {**DEFAULT_LOGGING_CONFIG}

config["formatters"]["vllm_tgis_adapter"] = DEFAULT_LOGGING_CONFIG["formatters"]["vllm"]

handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
handler_config["formatter"] = "vllm_tgis_adapter"
config["handlers"]["vllm_tgis_adapter"] = handler_config

logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
logger_config["handlers"] = ["vllm_tgis_adapter"]
config["loggers"]["vllm_tgis_adapter"] = logger_config

logging.config.dictConfig(config)
