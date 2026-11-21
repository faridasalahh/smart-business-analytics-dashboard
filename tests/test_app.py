import re
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as main


def setup_function():
    main.sessions.clear()


def test_dashboard_and_health_routes(monkeypatch):
    monkeypatch.setattr(main, "check_llm_health", lambda: "offline")
    with TestClient(main.app) as client:
        page = client.get("/")
        health = client.get("/api/health")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_rejects_untrusted_host():
    with TestClient(main.app) as client:
        response = client.get("/api/health", headers={"host": "attacker.example"})

    assert response.status_code == 400


def test_rejects_cross_origin_browser_writes():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/upload",
            headers={"origin": "https://attacker.example"},
            files=[("files", ("sample.csv", b"name,value\\nA,1\\n", "text/csv"))],
        )

    assert response.status_code == 403


def test_upload_sanitizes_filename_and_stores_inside_session(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(main, "UPLOAD_DIR", str(upload_root))

    with TestClient(main.app) as client:
        response = client.post(
            "/api/upload",
            files=[("files", ("../../sample.csv", b"name,value\\nA,1\\nB,2\\n", "text/csv"))],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["files"] == ["sample.csv"]

    session_id = payload["session_id"]
    stored_path = Path(main.sessions[session_id]["files"]["sample.csv"]).resolve()
    assert stored_path.parent == (upload_root / session_id).resolve()
    assert re.fullmatch(r"[0-9a-f]{32}_sample\\.csv", stored_path.name)
    assert stored_path.read_bytes().startswith(b"name,value")


def test_rejects_unsupported_upload_type(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path / "uploads"))

    with TestClient(main.app) as client:
        response = client.post(
            "/api/upload",
            files=[("files", ("payload.exe", b"not a dataset", "application/octet-stream"))],
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_expired_session_removes_runtime_files(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    charts = tmp_path / "charts"
    monkeypatch.setattr(main, "UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(main, "CHART_DIR", str(charts))
    (uploads / "expired").mkdir(parents=True)
    (charts / "expired").mkdir(parents=True)
    main.sessions["expired"] = {"_last_access": 0, "files": {}}

    assert main.get_session("expired") is None
    assert "expired" not in main.sessions
    assert not (uploads / "expired").exists()
    assert not (charts / "expired").exists()
