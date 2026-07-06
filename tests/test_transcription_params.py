"""get_timestamp_granularities: accepted combinations and 422 rejection."""

import pytest
from fastapi import HTTPException

from whisperx_api_server.routers.transcriptions import get_timestamp_granularities


def test_none_defaults_to_segment():
    assert get_timestamp_granularities(None) == ["segment"]


@pytest.mark.parametrize(
    "value",
    [[], ["segment"], ["word"], ["word", "segment"], ["segment", "word"]],
)
def test_valid_combinations_pass_through(value):
    assert get_timestamp_granularities(value) == value


@pytest.mark.parametrize("value", [["bogus"], ["word", "word"], ["segment", "segment"]])
def test_invalid_combinations_raise_422(value):
    with pytest.raises(HTTPException) as exc:
        get_timestamp_granularities(value)
    assert exc.value.status_code == 422
