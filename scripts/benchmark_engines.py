"""compare browser move selection using the same model and Stockfish oracle"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import shlex
import statistics
import subprocess

import chess
import chess.engine
import chess.pgn


ROOT = pathlib.Path(__file__).resolve().parents[1]


class BrowserEngine:
    def __init__(self, source):
        env = dict(os.environ, CHESS_SOURCE=str(source), NODE_PATH=str(ROOT / "node_modules"))
        self.process = subprocess.Popen(
            ["node", str(ROOT / "scripts/engine_bridge.cjs")],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def move(self, board, mode, options=None):
        request = {"fen": board.fen(en_passant="fen"), "mode": mode, "options": options or {}}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("browser engine exited before producing a move")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(result["error"])
        move = chess.Move.from_uci(result["move"])
        if move not in board.legal_moves:
            raise AssertionError(f"illegal {mode} move {move} in {board.fen()}")
        return move, result

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=20)


def model_hash():
    directory = ROOT / "public/nn"
    manifest = json.loads((directory / "model.json").read_text())
    files = ["model.json"] + [name for group in manifest["weightsManifest"] for name in group["paths"]]
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def source_hash(source):
    digest = hashlib.sha256()
    files = [source / "src/rules/chessRules.js"] + sorted(
        path for path in (source / "src/ai").glob("*.js") if not path.name.endswith(".test.js")
    )
    for path in files:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_positions(count):
    samples = json.loads((ROOT / "ml/data/fixed_eval_set_v3.json").read_text())
    fens = sorted({
        item["fen"] for item in samples
        if item.get("source") == "stockfish" and abs(float(item.get("cp", 0))) < 800
        and not chess.Board(item["fen"]).is_game_over(claim_draw=True)
    })
    random.Random(20261009).shuffle(fens)
    return fens[:count]


def score_cp(info, color):
    return info["score"].pov(color).score(mate_score=10000)


def position_benchmark(before, after, oracle, fens, nodes, options):
    records = []
    for mode in ("mcts", "nn"):
        for index, fen in enumerate(fens):
            board = chess.Board(fen)
            oracle.configure({"Clear Hash": None})
            best = oracle.analyse(board, chess.engine.Limit(nodes=nodes))
            record = {"mode": mode, "fen": fen, "oracle_move": best["pv"][0].uci(), "oracle_cp": score_cp(best, board.turn)}
            for name, player in (("before", before), ("after", after)):
                move, details = player.move(board, mode, options)
                oracle.configure({"Clear Hash": None})
                chosen = oracle.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=[move])
                record[name] = {
                    **details, "cp": score_cp(chosen, board.turn),
                    "cp_loss": max(0, record["oracle_cp"] - score_cp(chosen, board.turn)),
                    "oracle_match": move == best["pv"][0],
                }
            records.append(record)
            print(f"{mode} position {index + 1}/{len(fens)}: before {record['before']['move']} loss {record['before']['cp_loss']}, after {record['after']['move']} loss {record['after']['cp_loss']}", flush=True)
    summary = {}
    for mode in ("mcts", "nn"):
        subset = [record for record in records if record["mode"] == mode]
        if not subset:
            continue
        summary[mode] = {name: {
            "mean_cp_loss": statistics.mean(record[name]["cp_loss"] for record in subset),
            "median_cp_loss": statistics.median(record[name]["cp_loss"] for record in subset),
            "blunders_over_200cp": sum(record[name]["cp_loss"] > 200 for record in subset),
            "oracle_matches": sum(record[name]["oracle_match"] for record in subset),
            "mean_ms": statistics.mean(record[name]["elapsedMs"] for record in subset),
        } for name in ("before", "after")}
    return {"summary": summary, "positions": records}


def play_games(before, after, pairs, max_plies, options):
    openings = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "8/5pk1/5pp1/8/3P4/2P3P1/5PK1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/4K3/8/R7 w - - 0 1",
    ]
    records = []
    pgns = []
    for mode in ("mcts", "nn"):
        for pair in range(pairs):
            for after_color in (chess.WHITE, chess.BLACK):
                board = chess.Board(openings[pair % len(openings)])
                game = chess.pgn.Game()
                game.setup(board)
                game.headers["White"] = f"{mode} {'after' if after_color else 'before'}"
                game.headers["Black"] = f"{mode} {'before' if after_color else 'after'}"
                game.headers["Event"] = "same model engine comparison"
                node = game
                details = []
                for ply in range(max_plies):
                    if board.is_game_over(claim_draw=True):
                        break
                    name, player = ("after", after) if board.turn == after_color else ("before", before)
                    move, result = player.move(board, mode, options)
                    details.append({"player": name, "ply": ply, **result})
                    board.push(move)
                    node = node.add_variation(move)
                    if (ply + 1) % 10 == 0:
                        print(f"{mode} pair {pair + 1} after {'white' if after_color else 'black'}: {ply + 1} plies", flush=True)
                outcome = board.outcome(claim_draw=True)
                game.headers["Result"] = board.result(claim_draw=True)
                termination = outcome.termination.name if outcome else "PLY_LIMIT"
                game.headers["Termination"] = termination
                score = None if outcome is None else 0.5 if outcome.winner is None else int(outcome.winner == after_color)
                records.append({"mode": mode, "pair": pair, "after_color": "white" if after_color else "black", "score": score, "termination": termination, "plies": len(details), "moves": details})
                pgns.append(str(game))
                print(f"{mode} game result {game.headers['Result']} {termination}", flush=True)
    return records, "\n\n".join(pgns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=pathlib.Path, required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--oracle-nodes", type=int, default=20000)
    parser.add_argument("--game-pairs", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--time-ms", type=int, default=2000)
    parser.add_argument("--game-time-ms", type=int, default=2000)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    options = {"timeMs": args.time_ms, "tacticalTimeMs": max(1, int(args.time_ms * 0.6))}
    before = BrowserEngine(args.baseline_source.resolve())
    after = BrowserEngine(ROOT)
    try:
        with chess.engine.SimpleEngine.popen_uci(shlex.split(args.stockfish)) as oracle:
            oracle.configure({name: value for name, value in {"Threads": 1, "Hash": 16}.items() if name in oracle.options})
            result = position_benchmark(before, after, oracle, select_positions(args.positions), args.oracle_nodes, options)
            result["oracle"] = oracle.id
        if args.game_pairs:
            game_options = {"timeMs": args.game_time_ms, "tacticalTimeMs": max(1, int(args.game_time_ms * 0.6))}
            result["games"], pgn = play_games(before, after, args.game_pairs, args.max_plies, game_options)
            args.output.with_suffix(".pgn").write_text(pgn)
        result["model_sha256"] = model_hash()
        result["source_sha256"] = {"before": source_hash(args.baseline_source), "after": source_hash(ROOT)}
        result["baseline_source_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.baseline_source, text=True).strip() if (args.baseline_source / ".git").exists() else None
        result["source_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest()
        result["settings"] = vars(args) | {"baseline_source": str(args.baseline_source), "output": str(args.output)}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2), flush=True)
    finally:
        before.close()
        after.close()


if __name__ == "__main__":
    main()
