import httpx
import pytest

from arxiv_rag.config import Settings
from arxiv_rag.ingest.client import ArxivClient, ArxivHTTPError

pytestmark = pytest.mark.unit

OK_XML = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-02-01T00:00:00Z</updated>
    <published>2024-02-01T00:00:00Z</published>
    <title>Ok</title>
    <summary>Abstract.</summary>
    <author><name>Author</name></author>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""


class FakeClock:
    """Deterministic time: `sleep` records the wait and advances the clock."""

    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def make_client(handler, *, sleep_seconds=3.0, max_retries=2):
    clock = FakeClock()
    client = ArxivClient(
        Settings(
            _env_file=None,
            arxiv_sleep_seconds=sleep_seconds,
            arxiv_max_retries=max_retries,
        ),
        transport=httpx.MockTransport(handler),
        sleep_fn=clock.sleep,
        now_fn=clock.now,
    )
    return client, clock


def scripted(*steps):
    """Handler that serves queued responses / exceptions, recording requests."""
    steps = list(steps)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return handler, captured


def ok() -> httpx.Response:
    return httpx.Response(200, text=OK_XML)


def test_consecutive_requests_are_throttled():
    handler, _ = scripted(ok(), ok())
    client, clock = make_client(handler)
    client.fetch_page(["cs.AI"])
    client.fetch_page(["cs.AI"])
    assert clock.sleeps == [3.0]  # exactly one gap enforced before the 2nd request


def test_no_throttle_sleep_when_zero_interval():
    handler, _ = scripted(ok(), ok())
    client, clock = make_client(handler, sleep_seconds=0.0)
    client.fetch_page(["cs.AI"])
    client.fetch_page(["cs.AI"])
    assert clock.sleeps == []


def test_5xx_is_retried_then_succeeds():
    handler, captured = scripted(httpx.Response(503, text="slow down"), ok())
    client, clock = make_client(handler)
    records = client.fetch_page(["cs.AI"])
    assert [r.arxiv_id for r in records] == ["2401.00001"]
    assert len(captured) == 2
    assert clock.sleeps == [3.0]  # wait = max(backoff 1s, remaining throttle gap)


def test_retry_after_header_is_honored():
    handler, captured = scripted(
        httpx.Response(429, text="too many requests", headers={"Retry-After": "7"}),
        ok(),
    )
    client, clock = make_client(handler)
    client.fetch_page(["cs.AI"])
    assert len(captured) == 2
    assert clock.sleeps == [7.0]


def test_gives_up_after_max_retries():
    handler, captured = scripted(
        httpx.Response(503, text="slow down"),
        httpx.Response(503, text="slow down"),
        httpx.Response(503, text="slow down"),
    )
    client, clock = make_client(handler, max_retries=2)
    with pytest.raises(ArxivHTTPError, match=r"503.*after 3 attempt"):
        client.fetch_page(["cs.AI"])
    assert len(captured) == 3
    assert clock.sleeps == [3.0, 3.0]


def test_client_errors_are_not_retried():
    handler, captured = scripted(httpx.Response(400, text="bad request"))
    client, clock = make_client(handler)
    with pytest.raises(ArxivHTTPError, match="400"):
        client.fetch_page(["cs.AI"])
    assert len(captured) == 1
    assert clock.sleeps == []


def test_transport_errors_are_retried():
    handler, captured = scripted(httpx.ConnectError("connection refused"), ok())
    client, _clock = make_client(handler)
    records = client.fetch_page(["cs.AI"])
    assert [r.arxiv_id for r in records] == ["2401.00001"]
    assert len(captured) == 2
