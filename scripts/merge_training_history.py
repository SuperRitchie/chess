import json
import pathlib
import sys


def read_records(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def merge_records(current: list[dict], artifact: list[dict], limit: int = 365) -> list[dict]:
    by_timestamp = {}
    without_timestamp = []
    seen_legacy = set()
    for record in current + artifact:
        timestamp = record.get("timestamp_utc")
        if timestamp:
            by_timestamp[timestamp] = record
            continue
        fingerprint = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if fingerprint not in seen_legacy:
            seen_legacy.add(fingerprint)
            without_timestamp.append(record)
    merged = without_timestamp + [by_timestamp[key] for key in sorted(by_timestamp)]
    return merged[-limit:]


def main() -> None:
    current_path = pathlib.Path(sys.argv[1])
    artifact_path = pathlib.Path(sys.argv[2])
    merged = merge_records(read_records(current_path), read_records(artifact_path))
    current_path.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
