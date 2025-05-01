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

    # Track sync state for each channel
    cur.execute("""
      CREATE TABLE IF NOT EXISTS sync_state (
        channel_id TEXT PRIMARY KEY,
        last_sync_ts TEXT,
        last_sync_cursor TEXT,
        is_fully_synced BOOLEAN DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    """)

    # Track channels
    cur.execute("""
      CREATE TABLE IF NOT EXISTS channels (
        id       TEXT PRIMARY KEY,
        name     TEXT,
        raw_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    """)

    # Track messages with additional metadata
    cur.execute("""
      CREATE TABLE IF NOT EXISTS messages (
        channel_id TEXT,
        ts         TEXT,
        raw_json   TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (channel_id, ts)
      );
    """)

    # Track sync runs for debugging/auditing
    cur.execute("""
      CREATE TABLE IF NOT EXISTS sync_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        finished_at TIMESTAMP,
        status TEXT,
        error TEXT,
        channels_processed INTEGER DEFAULT 0,
        messages_processed INTEGER DEFAULT 0
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


def print_curl_command(method, url, headers, data=None, params=None, session_headers=None):
    """Print a curl command equivalent to the HTTP request."""

    # Add params
    if params:
        param_str = "&".join(f"{k}={v}" for k, v in params.items())

    cmd = ["curl"]
    cmd.append(f"'{url}?{param_str}'")

    # Add session headers first (like User-Agent and Cookie)
    if session_headers:
        for key, value in session_headers.items():
            cmd.append(f"-H '{key}: {value}'")

    # Add request-specific headers
    for key, value in headers.items():
        cmd.append(f"-H '{key}: {value}'")

    # Add method
    if method.upper() != "GET":
        cmd.append(f"-X {method}")

    # Add data
    if data:
        cmd.append(f"--data-raw $'{data}'")

    print("\nCurl command:")
    print(" ".join(cmd))
    print()


def fetch_all_channels(session, base_url, token, org_id, team_id, x_version_timestamp, db_conn, rate_limit, verbose,
                      client_req_id, browse_session_id):
    cur = db_conn.cursor()
    page = 1
    per_page = 50
    channels_processed = 0

    while True:
        # Create multipart form data
        boundary = "----WebKitFormBoundaryU4wEmw2oBAuXS3g9"
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": f"multipart/form-data; boundary={boundary}",
            "origin": "https://app.slack.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site"
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
        form_data.append('Content-Disposition: form-data; name="query"')
        form_data.append("")
        form_data.append("")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="page"')
        form_data.append("")
        form_data.append(str(page))

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="client_req_id"')
        form_data.append("")
        form_data.append(client_req_id)

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="browse_session_id"')
        form_data.append("")
        form_data.append(browse_session_id)

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="extracts"')
        form_data.append("")
        form_data.append("0")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="highlight"')
        form_data.append("")
        form_data.append("0")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="extra_message_data"')
        form_data.append("")
        form_data.append("0")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="no_user_profile"')
        form_data.append("")
        form_data.append("1")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="count"')
        form_data.append("")
        form_data.append(str(per_page))

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="file_title_only"')
        form_data.append("")
        form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="query_rewrite_disabled"')
        form_data.append("")
        form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="include_files_shares"')
        form_data.append("")
        form_data.append("1")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="browse"')
        form_data.append("")
        form_data.append("standard")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="search_context"')
        form_data.append("")
        form_data.append("desktop_channel_browser")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="max_filter_suggestions"')
        form_data.append("")
        form_data.append("10")

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

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="exclude_my_channels"')
        form_data.append("")
        form_data.append("0")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="search_only_team"')
        form_data.append("")
        form_data.append(team_id)

        # form_data.append(f"--{boundary}")
        # form_data.append('Content-Disposition: form-data; name="search_only_my_channels"')
        # form_data.append("")
        # form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="search_recently_left_channels"')
        form_data.append("")
        form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="search_recently_joined_channels"')
        form_data.append("")
        form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="_x_reason"')
        form_data.append("")
        form_data.append("browser-query")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="_x_mode"')
        form_data.append("")
        form_data.append("online")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="_x_sonic"')
        form_data.append("")
        form_data.append("true")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="_x_app_name"')
        form_data.append("")
        form_data.append("client")

        form_data.append(f"--{boundary}--")

        # Join with CRLF
        data = "\r\n".join(form_data)

        # Add query parameters to URL
        url = f"{base_url}/api/search.modules.channels"
        params = {
            "slack_route": f"{org_id}%3A{org_id}",
            "_x_version_ts": f"{x_version_timestamp}",
            "_x_frontend_build_type": "current",
            "_x_desktop_ia": "4",
            "_x_gantry": "true",
            "fp": "c7",
            "_x_num_retries": "0"
        }

        if verbose:
            logging.debug(f"POST {url}  page={page}")
            # print_curl_command("POST", url, headers, data, params=params, session_headers=session.headers)

        r = session.post(url, headers=headers, data=data, params=params)
        r.raise_for_status()
        j = r.json()
        # adjust this if your workspace returns under a different key
        items = j.get("items")
        if verbose:
            logging.debug(f"Got {len(items)} channels on page {page}")
        if not items:
            break
        for ch in items:
            ch_id = ch.get("id") or ch.get("channel", {}).get("id")
            ch_name = ch.get("name") or ch.get("channel", {}).get("name")
            raw = json.dumps(ch, separators=(",", ":"))

            # Update channel info and reset sync state if needed
            cur.execute("""
                INSERT OR REPLACE INTO channels (id, name, raw_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (ch_id, ch_name, raw))

            # Initialize sync state if it doesn't exist
            cur.execute("""
                INSERT OR IGNORE INTO sync_state (channel_id, is_fully_synced)
                VALUES (?, 0)
            """, (ch_id,))

            channels_processed += 1

        db_conn.commit()
        if len(items) < per_page:
            break
        page += 1
        print(f"sleeping for {rate_limit} seconds")
        time.sleep(rate_limit)


def fetch_channel_history(session, base_url, token, db_conn, rate_limit, verbose):
    cur = db_conn.cursor()
    cur2 = db_conn.cursor()
    cur3 = db_conn.cursor()

    # Start a new sync run
    cur.execute("INSERT INTO sync_runs (status) VALUES ('in_progress')")
    sync_run_id = cur.lastrowid
    db_conn.commit()

    try:
        # Get channels that need syncing
        cur.execute("""
            SELECT c.id, c.name, s.last_sync_ts, s.last_sync_cursor
            FROM channels c
            LEFT JOIN sync_state s ON c.id = s.channel_id
            WHERE s.is_fully_synced = 0 OR s.is_fully_synced IS NULL
            ORDER BY s.last_sync_ts ASC NULLS FIRST
        """)

        channels_to_sync = cur.fetchall()
        channels_processed = 0
        messages_processed = 0

        for ch_id, ch_name, last_sync_ts, last_cursor in channels_to_sync:
            if verbose:
                logging.info(f"Fetching history for #{ch_name} ({ch_id})")
                if last_sync_ts:
                    logging.info(f"  Resuming from {last_sync_ts}")

            cursor = last_cursor
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

                # Process messages
                for m in msgs:
                    ts = m.get("ts")
                    raw = json.dumps(m, separators=(",", ":"))

                    # Use INSERT OR REPLACE to update existing messages
                    cur2.execute("""
                        INSERT OR REPLACE INTO messages
                        (channel_id, ts, raw_json, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (ch_id, ts, raw))

                    messages_processed += 1

                db_conn.commit()

                # Update sync state
                if msgs:
                    oldest_ts = min(m.get("ts") for m in msgs)
                    cur3.execute("""
                        INSERT OR REPLACE INTO sync_state
                        (channel_id, last_sync_ts, last_sync_cursor, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (ch_id, oldest_ts, cursor))
                    db_conn.commit()

                cursor = j.get("response_metadata", {}).get("next_cursor")
                if verbose:
                    logging.debug(f"  fetched {len(msgs)} msgs, next_cursor={cursor}")

                if not cursor:
                    # Mark channel as fully synced if we've reached the end
                    cur3.execute("""
                        UPDATE sync_state
                        SET is_fully_synced = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE channel_id = ?
                    """, (ch_id,))
                    db_conn.commit()
                    break

                time.sleep(rate_limit)

            channels_processed += 1

            # Update sync run progress
            cur.execute("""
                UPDATE sync_runs
                SET channels_processed = ?, messages_processed = ?
                WHERE id = ?
            """, (channels_processed, messages_processed, sync_run_id))
            db_conn.commit()

            time.sleep(rate_limit)

        # Mark sync run as complete
        cur.execute("""
            UPDATE sync_runs
            SET status = 'completed',
                finished_at = CURRENT_TIMESTAMP,
                channels_processed = ?,
                messages_processed = ?
            WHERE id = ?
        """, (channels_processed, messages_processed, sync_run_id))
        db_conn.commit()

    except Exception as e:
        # Log error and mark sync run as failed
        logging.error(f"Sync failed: {str(e)}")
        cur.execute("""
            UPDATE sync_runs
            SET status = 'failed',
                error = ?,
                finished_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (str(e), sync_run_id))
        db_conn.commit()
        raise


def main():
    p = argparse.ArgumentParser(
        description="Archive Slack channels via in-browser endpoints"
    )
    p.add_argument(
        "subdomain", help="Slack workspace subdomain (e.g. datavant.enterprise)"
    )
    p.add_argument(
        "--org", required=True, help="Which specific Slack org the cookie is signed into"
    )
    p.add_argument(
        "--x-version-timestamp", required=True,
        help="X-Version-Timestamp from the Slack homepage"
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
    p.add_argument(
        "--client-req-id",
        help="Client request ID for API calls"
    )
    p.add_argument(
        "--browse-session-id",
        help="Browse session ID for API calls"
    )
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
        session, base_url, token, args.org, args.team, args.x_version_timestamp, conn, args.rate_limit, args.verbose,
        args.client_req_id, args.browse_session_id
    )

    # Fetch history for each channel
    fetch_channel_history(session, base_url, token, conn, args.rate_limit, args.verbose)

    logging.info("Done! Archive stored in %s", DB_PATH)
    conn.close()


if __name__ == "__main__":
    main()
