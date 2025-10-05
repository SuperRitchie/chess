# ml/stockfish_eval.py
"""
Evaluate FENs with Stockfish (centipawns). Requires a local Stockfish binary.
Set STOCKFISH_PATH env var in GitHub Actions or locally.
Depth is tuned for speed vs quality.
"""
import os, pathlib, subprocess, json, shlex

IN_FEN = pathlib.Path("ml/data/positions.fen")
OUT_JSON = pathlib.Path("ml/data/labels.json")

STOCKFISH = os.environ.get("STOCKFISH_PATH", "stockfish")
DEPTH = int(os.environ.get("SF_DEPTH", "12"))

def sf_eval(fen):
    # Use UCI with a single "position fen" + "go depth N" call
    # Simpler: spawn stockfish per call (OK for small datasets)
    cmds = [
        (f"uci\nisready\n"),
        (f"position fen {fen}\n"),
        (f"go depth {DEPTH}\n"),
    ]
    p = subprocess.Popen([STOCKFISH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = ""
    try:
        for c in cmds:
            p.stdin.write(c)
        p.stdin.flush()
        # read until bestmove
        for line in p.stdout:
            out += line
            if line.startswith("bestmove"):
                break
    finally:
        p.kill()

    # parse last "info score cp X" or "info score mate M"
    score_cp = 0
    last_info = [l for l in out.splitlines() if l.startswith("info ")]
    for l in reversed(last_info):
        if " score mate " in l:
            # mate in N: map to a large value with sign
            try:
                m = int(l.split(" score mate ")[1].split()[0])
                score_cp = 100000 if m > 0 else -100000
                break
            except: pass
        if " score cp " in l:
            try:
                cp = int(l.split(" score cp ")[1].split()[0])
                score_cp = cp
                break
            except: pass
    return score_cp

def main():
    data = []
    with IN_FEN.open() as f:
        fens = [line.strip() for line in f if line.strip()]
    for i, fen in enumerate(fens):
        sc = sf_eval(fen)
        data.append({"fen": fen, "cp": sc})
        if (i+1) % 200 == 0:
            print(f"eval {i+1}/{len(fens)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data), encoding="utf-8")
    print("wrote", OUT_JSON)

if __name__ == "__main__":
    main()
