from pathlib import Path

from tests.pdf_fixture import make_pdf


def test_health_starts_empty(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 0


def test_ingest_then_ask(client):
    files = [
        (
            "files",
            ("parrot.pdf", make_pdf(["The parrot is named Captain Flint."]), "application/pdf"),
        ),
        ("files", ("notes.md", b"# Notes\nThe ship is called the Hispaniola.\n", "text/markdown")),
        ("files", ("chart.png", b"\x89PNG\r\n\x1a\n fake image bytes", "image/png")),
    ]
    response = client.post("/api/ingest", files=files)
    assert response.status_code == 200, response.text
    report = response.json()
    assert sorted(report["files"]) == ["chart.png", "notes.md", "parrot.pdf"]
    assert report["chunks"] >= 3

    assert client.get("/api/health").json()["chunks"] >= 3
    sources = client.get("/api/sources").json()["sources"]
    assert {s["source"] for s in sources} == {"chart.png", "notes.md", "parrot.pdf"}

    response = client.post("/api/ask", json={"question": "What is the parrot's name?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "fake answer" in body["answer"]
    assert body["sources"][0]["n"] == 1
    assert "latency_ms" in body
    assert "x-response-time-ms" in response.headers


def test_rejects_unsupported_file_type(client):
    response = client.post(
        "/api/ingest", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))]
    )
    assert response.status_code == 400


def test_rejects_short_question(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422


def test_ask_with_empty_index_does_not_call_model(client):
    response = client.post("/api/ask", json={"question": "anything at all?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_upload_filename_is_sanitized(client, settings):
    files = [("files", ("../../etc/passwd.md", b"root:x:0:0", "text/markdown"))]
    assert client.post("/api/ingest", files=files).status_code == 200
    assert (Path(settings.upload_dir) / "passwd.md").exists()
    assert not (Path(settings.upload_dir).parent / "etc").exists()
