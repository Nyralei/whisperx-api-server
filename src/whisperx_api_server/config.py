from enum import Enum

from pydantic import BaseModel, Field
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
    vad_model: str = Field(default=None)
    vad_options: dict = Field(default=None)
    cache: bool = Field(default=True)
    preload_model: bool = Field(default=False)
    local_files_only: bool = Field(default=False)
    download_root: str = Field(default=None)
    batch_size: int = Field(default=12)
    chunk_size: int = Field(default=30)


class AlignConfig(BaseModel):
    models: dict = Field(default_factory=dict)
    whitelist: list = Field(default_factory=list)
    cache: bool = Field(default=True)
    preload_model: bool = Field(default=False)
    preload_model_name: str = Field(default=None)
    local_files_only: bool = Field(default=False)


class DiarizeConfig(BaseModel):
    model: str = Field(default="pyannote/speaker-diarization-community-1")
    cache: bool = Field(default=True)
    preload_model: bool = Field(default=False)


class Config(BaseSettings):
    """
    Configuration for the application. Values can be set via environment variables.

    Pydantic will automatically handle mapping uppercased environment variables to the corresponding fields.
    To populate nested, the environment should be prefixed with the nested field name and an underscore. For example,
    the environment variable `LOG_LEVEL` will be mapped to `log_level`, `WHISPER__MODEL`(note the double underscore) to `whisper.model`, to set quantization to int8, use `WHISPER__COMPUTE_TYPE=int8`, etc.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    api_key: str | None = None

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

    cache_cleanup: bool = True

    audio_cleanup: bool = True

    hf_token: str = Field(alias="HF_TOKEN", default="")
