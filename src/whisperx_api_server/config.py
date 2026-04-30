from enum import Enum

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    VTT_JSON = "vtt_json"
    SRT = "srt"
    VTT = "vtt"
    AUD = "aud"  # Audacity


class MediaType(str, Enum):
    APPLICATION_JSON = "application/json"
    TEXT_PLAIN = "text/plain"
    TEXT_VTT = "text/vtt"


class Language(str, Enum):
    AF = "af"
    AM = "am"
    AR = "ar"
    AS = "as"
    AZ = "az"
    BA = "ba"
    BE = "be"
    BG = "bg"
    BN = "bn"
    BO = "bo"
    BR = "br"
    BS = "bs"
    CA = "ca"
    CS = "cs"
    CY = "cy"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    EU = "eu"
    FA = "fa"
    FI = "fi"
    FO = "fo"
    FR = "fr"
    GL = "gl"
    GU = "gu"
    HA = "ha"
    HAW = "haw"
    HE = "he"
    HI = "hi"
    HR = "hr"
    HT = "ht"
    HU = "hu"
    HY = "hy"
    ID = "id"
    IS = "is"
    IT = "it"
    JA = "ja"
    JW = "jw"
    KA = "ka"
    KK = "kk"
    KM = "km"
    KN = "kn"
    KO = "ko"
    LA = "la"
    LB = "lb"
    LN = "ln"
    LO = "lo"
    LT = "lt"
    LV = "lv"
    MG = "mg"
    MI = "mi"
    MK = "mk"
    ML = "ml"
    MN = "mn"
    MR = "mr"
    MS = "ms"
    MT = "mt"
    MY = "my"
    NE = "ne"
    NL = "nl"
    NN = "nn"
    NO = "no"
    OC = "oc"
    PA = "pa"
    PL = "pl"
    PS = "ps"
    PT = "pt"
    RO = "ro"
    RU = "ru"
    SA = "sa"
    SD = "sd"
    SI = "si"
    SK = "sk"
    SL = "sl"
    SN = "sn"
    SO = "so"
    SQ = "sq"
    SR = "sr"
    SU = "su"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TG = "tg"
    TH = "th"
    TK = "tk"
    TL = "tl"
    TR = "tr"
    TT = "tt"
    UK = "uk"
    UR = "ur"
    UZ = "uz"
    VI = "vi"
    YI = "yi"
    YO = "yo"
    YUE = "yue"
    ZH = "zh"

# https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md


class Quantization(str, Enum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT8_BFLOAT16 = "int8_bfloat16"
    INT8_FLOAT32 = "int8_float32"
    INT16 = "int16"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    DEFAULT = "default"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class VadMethod(str, Enum):
    SILERO = "silero"
    PYANNOTE = "pyannote"


class WhisperConfig(BaseModel):
    """
    Default Huggingface model to use for transcription. Note, the model must support being ran using CTranslate2.
    This model will be used if no model is specified in the request.

    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    model: str = Field(default="large-v3")
    inference_device: Device = Field(default=Device.AUTO)
    device_index: int | list[int] = Field(default=0)
    compute_type: Quantization = Field(default=Quantization.DEFAULT)
    cpu_threads: int = Field(default=0)
    num_workers: int = Field(default=1)
    vad_method: VadMethod = Field(default=VadMethod.PYANNOTE)
    # Custom VAD model instance to assign directly. If set, vad_method is ignored.
    vad_model: str = Field(default=None)
    # Overrides for VAD defaults: chunk_size (s), vad_onset (0.0–1.0), vad_offset (0.0–1.0).
    # Defaults: {"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363}
    # Example: WHISPER__VAD_OPTIONS='{"vad_onset": 0.4, "vad_offset": 0.3}'
    vad_options: dict = Field(default=None)
    cache: bool = Field(default=True)
    preload_model: bool = Field(default=False)
    local_files_only: bool = Field(default=False)
    download_root: str = Field(default=None)
    batch_size: int = Field(default=12)
    chunk_size: int = Field(default=30)


class AlignConfig(BaseModel):
    # Override alignment models per language. Keys are ISO-639-1 language codes;
    # values are Hugging Face model names or local paths.
    # The special key "multilingual" applies one model to all languages regardless of code.
    # Example: ALIGNMENT__MODELS='{"ru": "bond005/wav2vec2-large-ru-golos", "multilingual": "voidful/wav2vec2-xlsr-multilingual-56"}'
    models: dict = Field(default_factory=dict)
    # Language codes to keep loaded in the alignment cache simultaneously.
    # Any model for a language not in this list is evicted when a new language is loaded.
    # Empty list = keep all loaded (unbounded cache).
    # Example: ALIGNMENT__WHITELIST='["ru", "en"]'
    whitelist: list = Field(default_factory=list)
    cache: bool = Field(default=True)
    preload_model: bool = Field(default=False)
    preload_model_name: str = Field(default=None)
    local_files_only: bool = Field(default=False)


class DiarizeConfig(BaseModel):
    model: str = Field(default="pyannote/speaker-diarization-community-1")
    cache: bool = Field(default=True)
    preload_model: bool = Field(default=False)


class BackendsConfig(BaseModel):
    transcription: str = Field(default="whisperx")
    alignment: str = Field(default="whisperx")
    diarization: str = Field(default="whisperx")


class DistributedMode(str, Enum):
    DIRECT = "direct"
    KAFKA = "kafka"


class KafkaConfig(BaseModel):
    bootstrap_servers: str = Field(default="localhost:9092")
    request_topic: str = Field(default="transcription-requests")
    reply_topic: str = Field(default="transcription-replies")
    consumer_group_worker: str = Field(default="whisperx-worker")
    reply_timeout_seconds: float = Field(default=3600.0)
    max_poll_interval_ms: int = Field(default=600_000)
    # Maximum number of jobs waiting for a reply (0 = unlimited).
    # Requests beyond this limit are rejected with HTTP 503.
    max_pending_jobs: int = Field(default=0)
    # Must match broker KAFKA_MESSAGE_MAX_BYTES (default 50 MiB).
    max_message_bytes: int = Field(default=52428800)


class S3Config(BaseModel):
    endpoint_url: str = Field(default="http://localhost:9000")
    access_key_id: str = Field(default="minioadmin")
    secret_access_key: str = Field(default="minioadmin", repr=False)
    bucket: str = Field(default="whisperx-audio")
    region: str = Field(default="us-east-1")
    delete_after_download: bool = Field(default=True)
    # Lifecycle expiry for objects in the bucket (days). 0 = disabled.
    object_expiry_days: int = Field(default=1)


class MetricsConfig(BaseModel):
    # When false, no observability code is loaded and /metrics route is not registered.
    # Set METRICS_ENABLED=true to enable. Requires the optional [metrics] extra
    # (pip install "whisperx-api-server[metrics]"). In direct (non-Kafka) mode,
    # METRICS_ENABLED=true assumes --workers 1 (per-app CollectorRegistry does not
    # share across worker processes).
    enabled: bool = Field(default=False)
    # Seconds between pynvml GPU polls.
    gpu_poll_interval: int = Field(default=15)
    # Port for the worker /metrics HTTP server (prometheus_client.start_http_server).
    # Each worker replica sets METRICS__WORKER_PORT to a unique value when scraped.
    # Used in DistributedMode.KAFKA to expose GPU metrics from worker processes that are not running the main ASGI server. Ignored in DistributedMode.DIRECT.
    worker_port: int = Field(default=9091)


class Config(BaseSettings):
    """
    Configuration for the application. Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__MODEL` (note the double underscore)
    to `whisper.model`, `BACKENDS__TRANSCRIPTION` to `backends.transcription`, and to set quantization to int8
    use `WHISPER__COMPUTE_TYPE=int8`, etc.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    api_key: str | None = Field(default=None, repr=False)

    api_keys_file: str | None = None

    log_level: str = "INFO"

    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(alias="UVICORN_PORT", default=8000)
    allow_origins: list[str] | None = None

    default_language: Language | None = None

    default_response_format: ResponseFormat = ResponseFormat.JSON

    whisper: WhisperConfig = WhisperConfig()

    alignment: AlignConfig = AlignConfig()

    diarization: DiarizeConfig = DiarizeConfig()

    backends: BackendsConfig = BackendsConfig()

    cache_cleanup: bool = True

    audio_cleanup: bool = True

    # Maximum number of concurrent GPU transcriptions (0 = unlimited, only applies when CUDA is available)
    max_concurrent_transcriptions: int = 1

    # Thread-pool workers for async I/O (audio load/save)
    io_executor_workers: int = 4

    hf_token: str = Field(alias="HF_TOKEN", default="", repr=False)

    mode: DistributedMode = Field(default=DistributedMode.DIRECT)

    kafka: KafkaConfig = KafkaConfig()

    s3: S3Config = S3Config()

    metrics: MetricsConfig = MetricsConfig()

    @model_validator(mode="before")
    @classmethod
    def _propagate_metrics_enabled(cls, values: dict) -> dict:
        """Map flat METRICS_ENABLED env var to metrics.enabled.

        pydantic-settings env_nested_delimiter='__' maps METRICS__ENABLED → metrics.enabled
        automatically, but the conventional shorter form METRICS_ENABLED is also supported
        here as an alias for operator convenience.  METRICS__ENABLED takes precedence if
        both are set.
        """
        import os

        flat_val = os.environ.get("METRICS_ENABLED")
        if flat_val is not None and isinstance(values, dict):
            metrics = values.get("metrics", {})
            if isinstance(metrics, dict) and "enabled" not in metrics:
                metrics["enabled"] = flat_val
                values["metrics"] = metrics
        return values
