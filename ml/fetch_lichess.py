# ml/fetch_lichess.py
"""
Download recent games from Lichess in PGN format.
You can filter by time controls or top users to curb volume.
"""
import os, time, requests, pathlib

OUT_PGN = pathlib.Path("ml/data/games.pgn")
OUT_PGN.parent.mkdir(parents=True, exist_ok=True)

# Example: public Lichess export for a pool of strong users
USERS = ["drnykterstein", "tsmftxh", "alireza2003", "rebeccaharris", "crew64"]
# Rapid + blitz only
PARAMS = {"max": 1000, "perfType": "blitz,rapid", "analysed": "false"}

def fetch_user(u):
    url = f"https://lichess.org/api/games/user/{u}"
    r = requests.get(url, params=PARAMS, headers={"Accept": "application/x-chess-pgn"})
    r.raise_for_status()
    return r.text

def main():
    with OUT_PGN.open("w", encoding="utf-8") as f:
        for u in USERS:
            print("Fetching", u)
            try:
                pgn = fetch_user(u)
                f.write(pgn.strip() + "\n\n")
                time.sleep(1.0)  # be gentle
            except Exception as e:
                print("Error", u, e)

if __name__ == "__main__":
    main()
