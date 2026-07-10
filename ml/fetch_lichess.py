# ml/fetch_lichess.py
"""Download recent public Lichess games with retries and validation."""
import os
import pathlib
import time

import requests

OUT_PGN = pathlib.Path("ml/data/games.pgn")
OUT_PGN.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_USERS = "drnykterstein,tsmftxh,alireza2003,rebeccaharris,crew64"
USERS = [user.strip() for user in os.environ.get("LICHESS_USERS", DEFAULT_USERS).split(",") if user.strip()]
MAX_GAMES = int(os.environ.get("LICHESS_MAX_GAMES", "1000"))
RETRIES = int(os.environ.get("LICHESS_FETCH_RETRIES", "3"))
TIMEOUT_SECONDS = float(os.environ.get("LICHESS_TIMEOUT_SECONDS", "30"))
PARAMS = {"max": MAX_GAMES, "perfType": "blitz,rapid", "analysed": "false"}
HEADERS = {
    "Accept": "application/x-chess-pgn",
    "User-Agent": "SuperRitchie-chess-training/1.0",
}


def fetch_user(session: requests.Session, username: str) -> str:
    url = f"https://lichess.org/api/games/user/{username}"
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, params=PARAMS, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                raise RuntimeError("empty PGN response")
            return text
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            print(f"[lichess] {username} attempt {attempt}/{RETRIES} failed: {exc}")
            if attempt < RETRIES:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"failed to fetch {username}: {last_error}")


def main():
    if not USERS:
        raise RuntimeError("LICHESS_USERS resolved to an empty user list")

    downloaded = []
    with requests.Session() as session:
        for username in USERS:
            print(f"[lichess] fetching {username}")
            try:
                downloaded.append(fetch_user(session, username))
            except RuntimeError as exc:
                print(f"[lichess] warning: {exc}")
            time.sleep(1.0)

    if not downloaded:
        raise RuntimeError("all Lichess downloads failed; refusing to overwrite the dataset")

    OUT_PGN.write_text("\n\n".join(downloaded) + "\n", encoding="utf-8")
    print(f"[lichess] wrote PGN data from {len(downloaded)}/{len(USERS)} users to {OUT_PGN}")


if __name__ == "__main__":
    main()
