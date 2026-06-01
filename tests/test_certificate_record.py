import json
from decimal import Decimal
from pathlib import Path


def test_stored_certificate_record():
    path = Path("runs/qstar_cert_roots_262144_target253635.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len(records) == 1
    record = records[0]
    assert record["certified"] is True
    assert record["status"] == "certified_upper_below_target"
    assert record["failed_panels"] == []
    assert record["panels"] == 262144
    assert record["precision"] == 120
    assert Decimal(str(record["bound_upper"])) <= Decimal("0.2536331090204145")
