from src.data_sources import _event_row, load_fixtures_with_status

def test_event_row_requires_both_teams():
    assert _event_row({"id":"1","competitions":[{"competitors":[]}]}, "Test") is None

def test_fixture_loader_returns_real_fixture_shape(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"events":[{"id":"123","date":"2026-11-13T12:00:00Z","status":{"type":{"name":"STATUS_SCHEDULED","detail":"Scheduled"}},"competitions":[{"competitors":[{"homeAway":"home","team":{"displayName":"Home FC"}},{"homeAway":"away","team":{"displayName":"Away FC"}}]}] }]}
    monkeypatch.setattr("src.data_sources.requests.get", lambda *args, **kwargs: Response())
    fixtures, status = load_fixtures_with_status(["England Premier"], __import__("datetime").date(2026,11,13))
    assert len(fixtures) == 1
    assert fixtures.iloc[0]["fixture_id"] == "123"
    assert status["England Premier"]["ok"] is True
