# ml/stockfish_eval.py
"""evaluate uncached positions and expert move policies with Stockfish"""
import json
import math
import os
import pathlib

import chess
import chess.engine

from fen_utils import canonical_fen
from policy_map import POLICY_VERSION, move_to_index

IN_FEN = pathlib.Path("ml/data/positions.fen")
OUT_JSON = pathlib.Path("ml/data/labels.json")
REPLAY_JSON = pathlib.Path("ml/data/replay_buffer.json")
STOCKFISH = os.environ.get("STOCKFISH_PATH", "stockfish")
DEPTH = int(os.environ.get("SF_DEPTH", "12"))
MULTIPV = int(os.environ.get("SF_MULTIPV", "3"))
POLICY_TEMPERATURE_CP = float(os.environ.get("SF_POLICY_TEMPERATURE_CP", "120"))
MATE_SCORE = 100000


def valid_policy(item: dict) -> bool:
    policy = item.get("policy")
    if not isinstance(policy, list):
        return False
    for entry in policy:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            if float(entry[1]) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def read_cached_labels() -> dict[str, dict]:
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
            normalized_fen = canonical_fen(fen)
            cache[normalized_fen] = {
                "fen": normalized_fen,
                "cp": float(item.get("cp", 0.0)),
                "policy_version": int(item.get("policy_version", 1)),
                "policy": item.get("policy", []),
            }
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
                normalized_fen = canonical_fen(fen)
            except ValueError as exc:
                print(f"[stockfish] skipped malformed FEN: {exc}")
                continue
            if normalized_fen in seen:
                continue
            seen.add(normalized_fen)
            unique.append(normalized_fen)
    return unique


def analyse_position(engine: chess.engine.SimpleEngine, fen: str) -> dict:
    board = chess.Board(fen)
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=DEPTH),
        multipv=max(1, MULTIPV),
    )
    if isinstance(infos, dict):
        infos = [infos]

    candidates = []
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
        candidates.append((pv[0], float(score if score is not None else 0.0)))
    if not candidates:
        raise ValueError("Stockfish returned no principal variation")

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_score = candidates[0][1]
    temperature = max(1.0, POLICY_TEMPERATURE_CP)
    weights = [math.exp(max(-40.0, (score - best_score) / temperature)) for _, score in candidates]
    total = sum(weights)
    policy = [
        [move_to_index(move), weight / total]
        for (move, _), weight in zip(candidates, weights)
    ]
    return {
        "fen": canonical_fen(fen),
        "cp": best_score,
        "policy_version": POLICY_VERSION,
        "policy": policy,
    }


def main():
    fens = read_unique_fens()
    cache = read_cached_labels()
    missing = [fen for fen in fens if fen not in cache or not valid_policy(cache[fen])]
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

    data = [cache[fen] for fen in fens if fen in cache and valid_policy(cache[fen])]
    if not data:
        raise RuntimeError("Stockfish evaluation produced no labels")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"[stockfish] wrote {len(data)} labels to {OUT_JSON}")


if __name__ == "__main__":
    main()
