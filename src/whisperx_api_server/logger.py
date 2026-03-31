import logging
import logging.config
import warnings


def setup_logger(log_level: str) -> None:
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level!r}. Must be one of {valid_levels}")

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
            "aiokafka": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "fsspec": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "lightning": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "pytorch_lightning": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
            "python_multipart": {
                "level": "WARNING",
                "handlers": ["default"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)

    # Suppress noisy but expected warnings from third-party libraries.
    # pyannote disables TF32 for reproducibility on every inference call — expected behaviour.
    warnings.filterwarnings(
        "ignore", message="TensorFloat-32", module="pyannote.*")
    # pyannote pooling layer emits this for very short segments with only one frame.
    warnings.filterwarnings(
        "ignore", message="std\\(\\): degrees of freedom", module="pyannote.*")
