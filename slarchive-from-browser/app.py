#!/usr/bin/env python3
import argparse
import requests
import sqlite3
import time
import json
import re
import logging
import sys

DB_PATH = "./db.sqlite3"


def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS channels (
        id       TEXT PRIMARY KEY,
        name     TEXT,
        raw_json TEXT
      );
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS messages (
        channel_id TEXT,
        ts         TEXT,
        raw_json   TEXT,
        PRIMARY KEY (channel_id, ts)
      );
    """)
    conn.commit()
    return conn


def extract_token(session, base_url, verbose=False):
    """Fetch homepage and extract api_token and team_id via regex."""
    url = f"{base_url}/"
    if verbose:
        logging.debug(f"GET {url}")
    r = session.get(url)
    r.raise_for_status()
    html = r.text

    token_m = re.search(r'"api_token":"([^"]+)"', html)
    if not token_m:
        logging.error("Failed to extract api_token from homepage")
        sys.exit(1)
    token = token_m.group(1)
    if verbose:
        logging.debug(f"Extracted token={token[:10]}")
    return token


def fetch_all_channels(session, base_url, token, team_id, db_conn, rate_limit, verbose):
    cur = db_conn.cursor()
    page = 1
    per_page = 50

    while True:
        # Create multipart form data
        boundary = "----WebKitFormBoundaryU4wEmw2oBAuXS3g9"
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Origin": "https://app.slack.com",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site"
        }

        # Build the multipart form data
        form_data = []
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="token"')
        form_data.append("")
        form_data.append(token)
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="module"')
        form_data.append("")
        form_data.append("channels")
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="page"')
        form_data.append("")
        form_data.append(str(page))
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="count"')
        form_data.append("")
        form_data.append(str(per_page))
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="search_only_team"')
        form_data.append("")
        form_data.append(team_id)
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="search_only_my_channels"')
        form_data.append("")
        form_data.append("false")
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="sort"')
        form_data.append("")
        form_data.append("name")
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="sort_dir"')
        form_data.append("")
        form_data.append("asc")
        
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="channel_type"')
        form_data.append("")
        form_data.append("exclude_archived")
        
        form_data.append(f"--{boundary}--")
        
        # Join with CRLF
        data = "\r\n".join(form_data)

        url = f"{base_url}/api/search.modules.channels"
        if verbose:
            logging.debug(f"POST {url}  page={page}")
        
        r = session.post(url, headers=headers, data=data)
        r.raise_for_status()
        j = r.json()
        # adjust this if your workspace returns under a different key
        items = j.get("results") or j.get("channels") or []
        if verbose:
            logging.debug(f"Got {len(items)} channels on page {page}")
        if not items:
            break
        for ch in items:
            ch_id = ch.get("id") or ch.get("channel", {}).get("id")
            ch_name = ch.get("name") or ch.get("channel", {}).get("name")
            raw = json.dumps(ch, separators=(",", ":"))
            cur.execute(
                "INSERT OR IGNORE INTO channels (id,name,raw_json) VALUES (?,?,?)",
                (ch_id, ch_name, raw),
            )
        db_conn.commit()
        if len(items) < per_page:
            break
        page += 1
        time.sleep(rate_limit)


def fetch_channel_history(session, base_url, token, db_conn, rate_limit, verbose):
    cur = db_conn.cursor()
    cur2 = db_conn.cursor()
    cur.execute("SELECT id,name FROM channels")
    for ch_id, ch_name in cur.fetchall():
        if verbose:
            logging.info(f"Fetching history for #{ch_name} ({ch_id})")
        cursor = None
        while True:
            params = {
                "token": token,
                "channel": ch_id,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            url = f"{base_url}/api/conversations.history"
            if verbose:
                logging.debug(f"GET {url}  cursor={cursor}")
            r = session.get(url, params=params)
            r.raise_for_status()
            j = r.json()
            msgs = j.get("messages", [])
            for m in msgs:
                ts = m.get("ts")
                raw = json.dumps(m, separators=(",", ":"))
                cur2.execute(
                    "INSERT OR IGNORE INTO messages (channel_id,ts,raw_json) VALUES"
                    " (?,?,?)",
                    (ch_id, ts, raw),
                )
            db_conn.commit()
            cursor = j.get("response_metadata", {}).get("next_cursor")
            if verbose:
                logging.debug(f"  fetched {len(msgs)} msgs, next_cursor={cursor}")
            if not cursor:
                break
            time.sleep(rate_limit)


def main():
    p = argparse.ArgumentParser(
        description="Archive Slack channels via in-browser endpoints"
    )
    p.add_argument(
        "subdomain", help="Slack workspace subdomain (e.g. datavant.enterprise)"
    )
    p.add_argument(
        "--team", required=True, help="Which specific Slack team the cookie is signed into"
    )
    p.add_argument(
        "--cookie", required=True, help="Your full Slack session cookie string"
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between HTTP requests",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = p.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")

    base_url = f"https://{args.subdomain}.slack.com"
    # build session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; SlackArchiver/1.0)",
        "Cookie": args.cookie,
    })

    # DB
    conn = setup_db()

    # Get internal token
    token = extract_token(session, base_url, args.verbose)

    # Enumerate channels
    fetch_all_channels(
        session, base_url, token, args.team, conn, args.rate_limit, args.verbose
    )

    # Fetch history for each channel
    fetch_channel_history(session, base_url, token, conn, args.rate_limit, args.verbose)

    logging.info("Done! Archive stored in %s", DB_PATH)
    conn.close()


if __name__ == "__main__":
    main()
