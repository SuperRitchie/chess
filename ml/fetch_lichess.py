# ml/fetch_lichess.py
"""download diverse public Lichess games with retries and validation"""
import datetime as dt
import os
import pathlib
import random
import time

import requests

OUT_PGN = pathlib.Path("ml/data/games.pgn")
OUT_PGN.parent.mkdir(parents=True, exist_ok=True)

FALLBACK_USERS = "alireza2003,rebeccaharris,crew64"
USER_OVERRIDE = [user.strip() for user in os.environ.get("LICHESS_USERS", "").split(",") if user.strip()]
LEADERBOARD_URL = "https://lichess.org/api/player/top/50/blitz"
FETCH_SEED = int(os.environ.get("LICHESS_FETCH_SEED", "42"))
USER_COUNT = int(os.environ.get("LICHESS_USER_COUNT", "10"))
MAX_GAMES = int(os.environ.get("LICHESS_MAX_GAMES", "300"))
HISTORY_WINDOWS = int(os.environ.get("LICHESS_HISTORY_WINDOWS", "52"))
HISTORY_STRIDE_DAYS = int(os.environ.get("LICHESS_HISTORY_STRIDE_DAYS", "7"))
RETRIES = int(os.environ.get("LICHESS_FETCH_RETRIES", "3"))
TIMEOUT_SECONDS = float(os.environ.get("LICHESS_TIMEOUT_SECONDS", "30"))
HEADERS = {
    "Accept": "application/x-chess-pgn",
    "User-Agent": "SuperRitchie-chess-training/1.0",
}


def history_cutoff_ms() -> int:
    window = FETCH_SEED % max(1, HISTORY_WINDOWS)
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=window * HISTORY_STRIDE_DAYS)
    return int(cutoff.timestamp() * 1000)


def discover_users(session: requests.Session) -> list[str]:
    if USER_OVERRIDE:
        return USER_OVERRIDE

    try:
        response = session.get(
            LEADERBOARD_URL,
            headers={**HEADERS, "Accept": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        users = response.json().get("users", [])
        names = [user.get("username") or user.get("id") for user in users]
        names = [name for name in names if name]
        random.Random(FETCH_SEED).shuffle(names)
        if names:
            selected = names[:max(1, USER_COUNT)]
            print(f"[lichess] selected {len(selected)} players from the live blitz leaderboard")
            return selected
    except (requests.RequestException, ValueError, AttributeError) as exc:
        print(f"[lichess] warning: leaderboard discovery failed: {exc}")

    fallback = [user.strip() for user in FALLBACK_USERS.split(",") if user.strip()]
    print(f"[lichess] using {len(fallback)} fallback players")
    return fallback


def fetch_user(session: requests.Session, username: str, until_ms: int) -> str:
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": MAX_GAMES,
        "until": until_ms,
        "perfType": "blitz,rapid",
        "analysed": "false",
        "clocks": "false",
        "evals": "false",
        "opening": "false",
    }
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            if response.status_code == 429 and attempt < RETRIES:
                retry_after = max(60, int(response.headers.get("Retry-After", "60")))
                print(f"[lichess] {username} rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue
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
    downloaded = []
    with requests.Session() as session:
        users = discover_users(session)
        if not users:
            raise RuntimeError("Lichess player list resolved to empty")
        until_ms = history_cutoff_ms()
        print(f"[lichess] historical cutoff {until_ms} from seed {FETCH_SEED}")
        for username in users:
            print(f"[lichess] fetching {username}")
            try:
                downloaded.append(fetch_user(session, username, until_ms))
            except RuntimeError as exc:
                print(f"[lichess] warning: {exc}")
            time.sleep(1.0)

    if not downloaded:
        raise RuntimeError("all Lichess downloads failed; refusing to overwrite the dataset")

    OUT_PGN.write_text("\n\n".join(downloaded) + "\n", encoding="utf-8")
    game_count = sum(text.count("[Event ") for text in downloaded)
    print(f"[lichess] wrote {game_count} games from {len(downloaded)}/{len(users)} players to {OUT_PGN}")


if __name__ == "__main__":
    main()
