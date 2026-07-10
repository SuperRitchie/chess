# ml/stockfish_eval.py
"""Evaluate only uncached FENs with one persistent Stockfish process."""
import json
import os
import pathlib

import chess
import chess.engine

IN_FEN = pathlib.Path("ml/data/positions.fen")
OUT_JSON = pathlib.Path("ml/data/labels.json")
REPLAY_JSON = pathlib.Path("ml/data/replay_buffer.json")
STOCKFISH = os.environ.get("STOCKFISH_PATH", "stockfish")
DEPTH = int(os.environ.get("SF_DEPTH", "12"))
MATE_SCORE = 100000


def read_cached_labels() -> dict[str, float]:
    if not REPLAY_JSON.exists():
        return {}
    try:
        items = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    cache = {}
    for item in items if isinstance(items, list) else []:
        fen = item.get("fen")
        if not fen:
            continue
        try:
            cache[fen] = float(item.get("cp", 0.0))
        except (TypeError, ValueError):
            continue
    return cache


def read_unique_fens() -> list[str]:
    seen = set()
    unique = []
    with IN_FEN.open(encoding="utf-8") as handle:
        for line in handle:
            fen = line.strip()
            if not fen or fen in seen:
                continue
            try:
                chess.Board(fen)
            except ValueError as exc:
                print(f"[stockfish] skipped malformed FEN: {exc}")
                continue
            seen.add(fen)
            unique.append(fen)
    return unique


def analyse_position(engine: chess.engine.SimpleEngine, fen: str) -> float:
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
    return float(score if score is not None else 0.0)


def main():
    fens = read_unique_fens()
    cache = read_cached_labels()
    missing = [fen for fen in fens if fen not in cache]
    print(f"[stockfish] positions {len(fens)}, cache hits {len(fens) - len(missing)}, new {len(missing)}")

    if missing:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        try:
            for index, fen in enumerate(missing, start=1):
                try:
                    cache[fen] = analyse_position(engine, fen)
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError, ValueError) as exc:
                    print(f"[stockfish] failed position {index}/{len(missing)}: {exc}")
                    continue
                if index % 200 == 0:
                    print(f"[stockfish] evaluated {index}/{len(missing)} new positions")
        finally:
            engine.quit()

    data = [{"fen": fen, "cp": cache[fen]} for fen in fens if fen in cache]
    if not data:
        raise RuntimeError("Stockfish evaluation produced no labels")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"[stockfish] wrote {len(data)} labels to {OUT_JSON}")


if __name__ == "__main__":
    main()
