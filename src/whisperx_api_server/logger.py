import logging
import logging.config


def setup_logger(log_level: str) -> None:
    assert log_level.upper() in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }, log_level

    logging_config = {
        "version": 1,  # required
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "use_colors": True
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "root": {
                "level": log_level.upper(),
                "handlers": ["default"],
            },
        },
    }

    logging.config.dictConfig(logging_config)