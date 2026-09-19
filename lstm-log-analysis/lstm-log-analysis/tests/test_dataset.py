"""Unit tests for dataset generation, parsing, and session grouping."""

from src.dataset import (
    parse_log_line,
    generate_synthetic_benchmark_dataset,
    HDFS_TEMPLATES,
)


def test_parse_valid_log_line():
    """Verify regex parser extracts structured attributes correctly."""
    sample_line = (
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: "
        "Receiving block blk_-160899968791986290 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    )
    parsed = parse_log_line(sample_line)
    assert parsed is not None
    assert parsed["Date"] == "081109"
    assert parsed["Time"] == "203518"
    assert parsed["Pid"] == 143
    assert parsed["Level"] == "INFO"
    assert parsed["Component"] == "dfs.DataNode$DataXceiver"
    assert parsed["BlockId"] == "blk_-160899968791986290"
    assert parsed["EventId"] == "E1"


def test_parse_invalid_log_line():
    """Verify unparseable text returns None."""
    invalid_line = "This is a random non-log string with no header."
    parsed = parse_log_line(invalid_line)
    assert parsed is None


def test_generate_synthetic_benchmark():
    """Verify benchmark data generator creates requested sessions and labels."""
    num_sessions = 50
    anomaly_ratio = 0.20
    raw_lines, labels = generate_synthetic_benchmark_dataset(
        num_sessions=num_sessions,
        anomaly_ratio=anomaly_ratio,
        seed=123,
    )
    assert len(labels) == num_sessions
    assert len(raw_lines) > num_sessions

    normal_count = sum(1 for s in labels if s["Label"] == "Normal")
    anomaly_count = sum(1 for s in labels if s["Label"] == "Anomaly")
    assert anomaly_count == int(num_sessions * anomaly_ratio)
    assert normal_count == num_sessions - anomaly_count


def test_hdfs_templates_integrity():
    """Verify every template has valid EventId, Regex, and Component."""
    assert len(HDFS_TEMPLATES) > 0
    event_ids = set()
    for tmpl in HDFS_TEMPLATES:
        assert tmpl["EventId"].startswith("E")
        assert tmpl["EventId"] not in event_ids
        event_ids.add(tmpl["EventId"])
        assert "Template" in tmpl
        assert "Regex" in tmpl
        assert "Component" in tmpl
