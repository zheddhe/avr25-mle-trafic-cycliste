#!/usr/bin/env python3
"""
Simple API load simulator focused on /predictions/{counter_id}.

- Authenticated requests (p_ok) → expect 200 and JSON payload.
- Unauthenticated requests (1 - p_ok) → expect 401/403 (counted as 4XX).
- Reports %200, %4XX, total RPS and predictions/s.

Usage example:
  python3 scripts/sim_api_req.py \
    --url http://localhost:8000 \
    --n 120 --p-ok 0.85 --user user1 --password user1 --limit 10
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import random
import time
import pytest
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

pytestmark = pytest.mark.integration

LOGGER = logging.getLogger("sim_api_req")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass
class Stats:
    ok_200: int = 0
    ok_2xx: int = 0
    err_4xx: int = 0
    err_5xx: int = 0
    other: int = 0
    predictions_returned: int = 0


def build_auth_header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def build_url(base_url: str, path: str) -> str:
    base = base_url[:-1] if base_url.endswith("/") else base_url
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def http_get(url: str, auth_header: Optional[str]) -> int | tuple[int, bytes]:
    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:  # network errors, timeouts, etc.
        LOGGER.debug("Network error for %s: %s", url, exc)
        return 0


def fetch_counters(url_base: str, auth_header: str) -> List[str]:
    url = build_url(url_base, "/counters")
    res = http_get(url, auth_header)
    if isinstance(res, tuple):
        status, body = res
    else:
        status, body = res, b""

    if status != 200:
        raise RuntimeError(
            f"Failed to fetch counters: HTTP {status} (need valid auth?)"
        )

    try:
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid /counters JSON: {exc}") from exc

    # /counters returns a list of {"id": "..."}
    counters = [item["id"] for item in data if "id" in item]
    if not counters:
        raise RuntimeError("No counters available from /counters.")
    return counters


def pick_counter(counters: List[str]) -> str:
    return random.choice(counters)


def build_predictions_url(
    url_base: str,
    counter_id: str,
    limit: int,
    offset: int,
) -> str:
    path = f"/predictions/{urllib.parse.quote(counter_id)}"
    qs = urllib.parse.urlencode({"limit": limit, "offset": offset})
    return f"{build_url(url_base, path)}?{qs}"


def run_request(
    url: str,
    with_auth: bool,
    auth_header: Optional[str],
) -> tuple[int, int]:
    """
    Returns (status_code, predictions_count_in_body_if_200_else_0)
    """
    header = auth_header if with_auth else None
    res = http_get(url, header)
    if isinstance(res, tuple):
        status, body = res
    else:
        return res, 0

    if status == 200:
        try:
            payload = json.loads(body.decode("utf-8"))
            items = payload.get("item", [])
            return status, int(len(items))
        except Exception as exc:
            LOGGER.debug("Invalid JSON for %s: %s", url, exc)
            return status, 0
    return status, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate traffic on /predictions/{counter_id}."
    )
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p-ok", type=float, default=0.8)
    parser.add_argument("--user", type=str, default="user1")
    parser.add_argument("--password", type=str, default="user1")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--counter-ids",
        type=str,
        default="",
        help="Comma-separated counter ids. If empty, auto-fetch via /counters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    auth_header = build_auth_header(args.user, args.password)

    if args.counter_ids.strip():
        counters = [c.strip() for c in args.counter_ids.split(",") if c.strip()]
    else:
        counters = fetch_counters(args.url, auth_header)

    LOGGER.info(
        "Loaded %d counters. Example: %s",
        len(counters),
        counters[:3],
    )

    stats = Stats()
    t0 = time.perf_counter()

    for _ in range(args.n):
        counter_id = pick_counter(counters)
        url = build_predictions_url(args.url, counter_id, args.limit, args.offset)
        with_auth = random.random() < args.p_ok
        status, n_pred = run_request(url, with_auth, auth_header)

        if status == 200:
            stats.ok_200 += 1
            stats.ok_2xx += 1
            stats.predictions_returned += n_pred
        elif 200 < status < 300:
            stats.ok_2xx += 1
        elif 400 <= status < 500:
            stats.err_4xx += 1
        elif 500 <= status < 600:
            stats.err_5xx += 1
        else:
            stats.other += 1

    elapsed = max(1e-9, time.perf_counter() - t0)
    rps = args.n / elapsed
    preds_per_sec = stats.predictions_returned / elapsed

    pct_200 = 100.0 * stats.ok_200 / max(1, args.n)
    pct_4xx = 100.0 * stats.err_4xx / max(1, args.n)

    # Final summary (stdout)
    print("=== sim_api_req summary ===")
    print(f"Base URL       : {args.url}")
    print(f"Counters       : {len(counters)}")
    print(f"Requests       : {args.n} "
          f"(elapsed {elapsed:.3f}s, ~{rps:.2f} rps)")
    print(f"HTTP 200       : {stats.ok_200}  ({pct_200:.1f}%)")
    print(f"HTTP 4XX       : {stats.err_4xx} ({pct_4xx:.1f}%)")
    print(f"HTTP 5XX       : {stats.err_5xx}")
    print(f"Other/Unknown  : {stats.other}")
    print(f"Predictions/s  : {preds_per_sec:.2f} "
          f"(total {stats.predictions_returned})")


if __name__ == "__main__":
    main()
