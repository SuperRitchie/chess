import json
import pathlib
import sys


def merge_records(current, incoming, limit=365):
    records = {}
    for record in current + incoming:
        key = record.get("timestamp_utc") or json.dumps(record, sort_keys=True)
        records[key] = record
    return sorted(records.values(), key=lambda item: item.get("timestamp_utc", ""))[-limit:]


def main():
    current = pathlib.Path(sys.argv[1])
    incoming = pathlib.Path(sys.argv[2])
    merged = merge_records(json.loads(current.read_text()), json.loads(incoming.read_text()))
    current.write_text(json.dumps(merged, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
