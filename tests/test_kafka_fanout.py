"""Reply/progress consumers must stay groupless.

A consumer group here is a leak, not merely overhead: fan-out requires one group
per replica, and nothing ever reads the committed offsets back (the position is
always "latest"), so every restart would strand another group on the coordinator
permanently.
"""

import pytest

from whisperx_api_server.config import KafkaConfig
from whisperx_api_server.kafka_client import _fanout_consumer

try:
    import aiokafka  # noqa: F401

    HAS_AIOKAFKA = True
except Exception:
    HAS_AIOKAFKA = False

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not HAS_AIOKAFKA, reason="aiokafka (kafka extras) not installed"
    ),
]


async def test_fanout_consumers_join_no_group():
    cfg = KafkaConfig()
    for topic in (cfg.reply_topic, cfg.progress_topic):
        consumer = _fanout_consumer(cfg, topic)
        assert consumer._group_id is None
        assert consumer._enable_auto_commit is False


async def test_fanout_consumer_starts_from_latest():
    """Replies already on the topic are not replayed into a fresh replica."""
    consumer = _fanout_consumer(KafkaConfig(), "transcription-replies")
    assert consumer._auto_offset_reset == "latest"
