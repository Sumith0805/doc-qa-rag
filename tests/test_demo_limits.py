"""Public-demo limits: upload size, library size, no deleting."""
import pytest

import config
import registry
import store


def test_oversized_file_is_rejected_in_demo_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DEMO_MAX_UPLOAD_MB", 1)
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (2 * 1024 * 1024))
    with pytest.raises(store.DemoLimitError, match="1 MB"):
        store.index_document(str(big), "big.txt")
    assert registry.list_documents() == []  # nothing half-registered


def test_library_size_is_capped_in_demo_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DEMO_MAX_DOCS", 1)
    registry.create_document("d1", "a.txt", "h1")
    registry.mark_ready("d1", pages=1, chunks=1)
    new = tmp_path / "new.txt"
    new.write_text("some new text", encoding="utf-8")
    with pytest.raises(store.DemoLimitError, match="full"):
        store.index_document(str(new), "new.txt")
    assert len(registry.list_documents()) == 1


def test_failed_documents_do_not_use_up_the_library(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DEMO_MAX_DOCS", 1)
    registry.create_document("d1", "a.txt", "h1")
    registry.mark_failed("d1", "boom")
    small = tmp_path / "ok.txt"
    small.write_text("fine", encoding="utf-8")
    store._check_demo_limits(str(small))  # does not raise


def test_deleting_is_disabled_in_demo_mode(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    with pytest.raises(store.DemoLimitError, match="disabled"):
        store.delete_document("anything")


def test_limits_are_not_enforced_outside_demo_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "DEMO_MAX_UPLOAD_MB", 0)
    monkeypatch.setattr(config, "DEMO_MAX_DOCS", 0)
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 1024)
    store._check_demo_limits(str(big))  # does not raise