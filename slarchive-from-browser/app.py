#!/usr/bin/env python3
import argparse
import requests
import sqlite3
import time
import json
import re
import logging
import sys
import os
import gzip
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

DB_PATH = "./db.sqlite3"


@dataclass
class SlackConfig:
    """Configuration for Slack API access."""

    subdomain: str
    org_id: str
    team_id: str
    x_version_timestamp: str
    cookie: str
    rate_limit: float = 1.0
    verbose: bool = False
    client_req_id: Optional[str] = None
    browse_session_id: Optional[str] = None


class DatabaseManager:
    """Manages database operations and compression."""

    CURRENT_VERSION = 2  # Increment this when schema changes

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self) -> None:
        """Establish database connection and setup schema."""
        self.conn = sqlite3.connect(self.db_path)
        self._setup_schema()
        self._check_and_migrate()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def get_user_display_name(self, user_id: str) -> str:
        """Get the best available display name for a user."""
        if not user_id:
            return "Unknown"

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT real_name, display_name, name
            FROM users
            WHERE id = ?
        """,
            (user_id,),
        )

        row = cur.fetchone()
        if not row:
            return f"<@{user_id}>"

        real_name, display_name, name = row

        # Return the first non-empty name in order of preference
        return display_name or real_name or name or f"<@{user_id}>"

    def _setup_schema(self) -> None:
        """Create database tables if they don't exist."""
        cur = self.conn.cursor()

        # Create version tracking table first
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS db_version (
                version INTEGER PRIMARY KEY,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )

        # Create all other tables
        tables = [
            self._get_sync_state_schema(),
            self._get_channels_schema(),
            self._get_messages_schema(),
            self._get_users_schema(),
            self._get_sync_runs_schema(),
            self._get_compressed_data_schema(),
        ]

        for schema in tables:
            cur.execute(schema)

        self.conn.commit()

    def _check_and_migrate(self) -> None:
        """Check database version and run migrations if needed."""
        cur = self.conn.cursor()

        # Get current version
        cur.execute("SELECT version FROM db_version ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        current_version = row[0] if row else 0

        if current_version < self.CURRENT_VERSION:
            self._run_migrations(current_version)

    def _run_migrations(self, from_version: int) -> None:
        """Run all necessary migrations."""
        if from_version < 2:
            self._migrate_to_v2()

        # Update version
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO db_version (version) VALUES (?)", (self.CURRENT_VERSION,)
        )
        self.conn.commit()

    def _migrate_to_v2(self) -> None:
        """Migrate from version 1 to version 2 (compressed data)."""
        logging.info("Starting migration to version 2 (compressed data)...")

        # Create compressed_data table if it doesn't exist
        cur = self.conn.cursor()
        cur.execute(self._get_compressed_data_schema())

        # Migrate channels
        logging.info("Migrating channels...")
        cur.execute("SELECT id, raw_json FROM channels WHERE raw_json IS NOT NULL")
        for ch_id, raw_json in cur.fetchall():
            if raw_json:
                self.store_compressed_data("channels", ch_id, raw_json)
                cur.execute(
                    "UPDATE channels SET raw_json = NULL WHERE id = ?", (ch_id,)
                )

        # Migrate messages
        logging.info("Migrating messages...")
        cur.execute(
            "SELECT channel_id, ts, raw_json FROM messages WHERE raw_json IS NOT NULL"
        )
        for ch_id, ts, raw_json in cur.fetchall():
            if raw_json:
                self.store_compressed_data("messages", f"{ch_id}_{ts}", raw_json)
                cur.execute(
                    "UPDATE messages SET raw_json = NULL WHERE channel_id = ? AND ts = ?",
                    (ch_id, ts),
                )

        # Migrate users
        logging.info("Migrating users...")
        cur.execute("SELECT id, raw_json FROM users WHERE raw_json IS NOT NULL")
        for user_id, raw_json in cur.fetchall():
            if raw_json:
                self.store_compressed_data("users", user_id, raw_json)
                cur.execute("UPDATE users SET raw_json = NULL WHERE id = ?", (user_id,))

        self.conn.commit()
        logging.info("Migration to version 2 completed successfully")

    def _get_sync_state_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS sync_state (
            channel_id TEXT PRIMARY KEY,
            last_sync_ts TEXT,
            last_sync_cursor TEXT,
            is_fully_synced BOOLEAN DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    def _get_channels_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS channels (
            id       TEXT PRIMARY KEY,
            name     TEXT,
            raw_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    def _get_messages_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS messages (
            channel_id TEXT,
            ts TEXT,
            thread_ts TEXT,
            user_id TEXT,
            subtype TEXT,
            client_msg_id TEXT,
            edited_ts TEXT,
            edited_user TEXT,
            reply_count INTEGER,
            reply_users_count INTEGER,
            latest_reply TEXT,
            is_locked BOOLEAN,
            has_files BOOLEAN,
            has_blocks BOOLEAN,
            raw_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (channel_id, ts)
        );
        """

    def _get_users_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS users (
            id         TEXT PRIMARY KEY,
            name       TEXT,
            real_name  TEXT,
            display_name TEXT,
            raw_json   TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

    def _get_sync_runs_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS sync_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP,
            status TEXT,
            error TEXT,
            channels_processed INTEGER DEFAULT 0,
            messages_processed INTEGER DEFAULT 0
        );
        """

    def _get_compressed_data_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS compressed_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            record_id TEXT NOT NULL,
            compressed_data BLOB NOT NULL,
            original_size INTEGER NOT NULL,
            compressed_size INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(table_name, record_id)
        );
        """

    def compress_data(self, data: str) -> bytes:
        """Compress data using gzip."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return gzip.compress(data)

    def decompress_data(self, compressed_data: bytes) -> Optional[str]:
        """Decompress gzip data."""
        if compressed_data is None:
            return None
        return gzip.decompress(compressed_data).decode("utf-8")

    def store_compressed_data(self, table_name: str, record_id: str, data: str) -> None:
        """Store compressed data in the compressed_data table."""
        if not data:
            return

        compressed = self.compress_data(data)
        original_size = len(data.encode("utf-8") if isinstance(data, str) else data)
        compressed_size = len(compressed)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO compressed_data
            (table_name, record_id, compressed_data, original_size, compressed_size, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (table_name, record_id, compressed, original_size, compressed_size),
        )
        self.conn.commit()

    def get_compressed_data(self, table_name: str, record_id: str) -> Optional[str]:
        """Retrieve and decompress data from the compressed_data table."""
        cur = self.conn.cursor()

        # First try compressed data
        cur.execute(
            """
            SELECT compressed_data
            FROM compressed_data
            WHERE table_name = ? AND record_id = ?
        """,
            (table_name, record_id),
        )

        row = cur.fetchone()
        if row:
            return self.decompress_data(row[0])

        # If not found in compressed_data, try raw_json (for backward compatibility)
        if table_name == "channels":
            cur.execute("SELECT raw_json FROM channels WHERE id = ?", (record_id,))
        elif table_name == "messages":
            ch_id, ts = record_id.split("_")
            cur.execute(
                "SELECT raw_json FROM messages WHERE channel_id = ? AND ts = ?",
                (ch_id, ts),
            )
        elif table_name == "users":
            cur.execute("SELECT raw_json FROM users WHERE id = ?", (record_id,))

        row = cur.fetchone()
        if row and row[0]:
            # If found in raw_json, migrate it to compressed_data
            self.store_compressed_data(table_name, record_id, row[0])
            return row[0]

        return None


class SlackClient:
    """Handles Slack API interactions."""

    def __init__(self, config: SlackConfig):
        self.config = config
        self.session = self._create_session()
        self.token = None

    def _create_session(self) -> requests.Session:
        """Create and configure requests session."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; SlackArchiver/1.0)",
                "Cookie": self.config.cookie,
            }
        )
        return session

    def _print_curl_command(
        self,
        method: str,
        url: str,
        headers: Dict,
        data: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> None:
        """Print a curl command equivalent to the HTTP request."""
        cmd = ["curl"]

        # Add params
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in params.items())
            cmd.append(f"'{url}?{param_str}'")
        else:
            cmd.append(f"'{url}'")

        # Add session headers first
        for key, value in self.session.headers.items():
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

    def initialize(self) -> None:
        """Initialize the client by extracting the token."""
        self.token = self._extract_token()

    def _extract_token(self) -> str:
        """Extract API token from Slack homepage."""
        url = f"https://{self.config.subdomain}.slack.com/"
        if self.config.verbose:
            logging.debug(f"GET {url}")

        r = self.session.get(url)
        r.raise_for_status()

        token_m = re.search(r'"api_token":"([^"]+)"', r.text)
        if not token_m:
            raise ValueError("Failed to extract api_token from homepage")

        token = token_m.group(1)
        if self.config.verbose:
            logging.debug(f"Extracted token={token[:10]}")
        return token


class ChannelManager:
    """Manages channel-related operations."""

    def __init__(self, db: DatabaseManager, slack: SlackClient):
        self.db = db
        self.slack = slack

    def fetch_all_channels(self) -> int:
        """Fetch all channels from Slack."""
        cur = self.db.conn.cursor()
        page = 1
        per_page = 50
        channels_processed = 0

        print("\nFetching channels...")

        while True:
            channels = self._fetch_channel_page(page, per_page)
            if not channels:
                break

            if self.slack.config.verbose:
                logging.debug(f"Got {len(channels)} channels on page {page}")
            else:
                print(f"Page {page}: {len(channels)} channels")

            # Process channels
            for ch in channels:
                ch_id = ch.get("id") or ch.get("channel", {}).get("id")
                ch_name = ch.get("name") or ch.get("channel", {}).get("name")

                # Update channel info and reset sync state if needed
                cur.execute(
                    """
                    INSERT OR REPLACE INTO channels (id, name, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                    (ch_id, ch_name),
                )

                # Initialize sync state if it doesn't exist
                cur.execute(
                    """
                    INSERT OR IGNORE INTO sync_state (channel_id, is_fully_synced)
                    VALUES (?, 0)
                """,
                    (ch_id,),
                )

                channels_processed += 1

            self.db.conn.commit()

            if len(channels) < per_page:
                break

            page += 1
            if self.slack.config.verbose:
                print(f"Sleeping for {self.slack.config.rate_limit} seconds")
            time.sleep(self.slack.config.rate_limit)

        print(f"\nFound {channels_processed} channels")
        return channels_processed

    def _fetch_channel_page(self, page: int, per_page: int) -> List[Dict]:
        """Fetch a single page of channels from Slack."""
        url = f"https://{self.slack.config.subdomain}.slack.com/api/search.modules.channels"

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
            "sec-fetch-site": "same-site",
        }

        # Build the multipart form data
        form_data = []
        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="token"')
        form_data.append("")
        form_data.append(self.slack.token)

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
        form_data.append(self.slack.config.client_req_id or "")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="browse_session_id"')
        form_data.append("")
        form_data.append(self.slack.config.browse_session_id or "")

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
        form_data.append(self.slack.config.team_id)

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

        # Add query parameters
        params = {
            "slack_route": f"{self.slack.config.org_id}%3A{self.slack.config.org_id}",
            "_x_version_ts": self.slack.config.x_version_timestamp,
            "_x_frontend_build_type": "current",
            "_x_desktop_ia": "4",
            "_x_gantry": "true",
            "fp": "c7",
            "_x_num_retries": "0",
        }

        if self.slack.config.verbose:
            logging.debug(f"POST {url}  page={page}")
            self.slack._print_curl_command("POST", url, headers, data=data, params=params)

        r = self.slack.session.post(url, headers=headers, data=data, params=params)
        r.raise_for_status()
        j = r.json()

        # adjust this if your workspace returns under a different key
        return j.get("items", [])


class MessageManager:
    """Manages message-related operations."""

    def __init__(self, db: DatabaseManager, slack: SlackClient):
        self.db = db
        self.slack = slack
        self._last_response = None

    def sync_all_channels(self) -> Tuple[int, int]:
        """Sync messages from all channels using a round-robin approach."""
        sync_cur = self.db.conn.cursor()  # For sync operations
        msg_cur = self.db.conn.cursor()   # For message operations

        # Start a new sync run
        sync_cur.execute("INSERT INTO sync_runs (status) VALUES ('in_progress')")
        sync_run_id = sync_cur.lastrowid
        self.db.conn.commit()

        try:
            # Get channels that need syncing
            sync_cur.execute(
                """
                SELECT c.id, c.name, s.last_sync_ts, s.last_sync_cursor, s.is_fully_synced
                FROM channels c
                LEFT JOIN sync_state s ON c.id = s.channel_id
                ORDER BY s.last_sync_ts ASC NULLS FIRST
            """
            )

            channels_to_sync = sync_cur.fetchall()
            total_channels = len(channels_to_sync)
            channels_processed = 0
            messages_processed = 0

            print(f"\nStarting sync of {total_channels} channels...")

            # Track sync state for each channel
            channel_states = {
                ch_id: {
                    'name': ch_name,
                    'cursor': last_cursor,
                    'messages': 0,
                    'pages': 0,
                    'is_complete': False,
                    'is_fully_synced': bool(is_fully_synced),
                    'latest_ts': None,
                    'oldest_ts': None,
                    'sync_direction': 'forward' if last_cursor else 'backward'
                }
                for ch_id, ch_name, _, last_cursor, is_fully_synced in channels_to_sync
            }

            # Calculate max channel name length for padding
            max_channel_length = max(len(state['name']) for state in channel_states.values())

            while any(not state['is_complete'] for state in channel_states.values()):
                for ch_id, state in channel_states.items():
                    if state['is_complete']:
                        continue

                    ch_name = state['name']
                    cursor = state['cursor']

                    if self.slack.config.verbose:
                        logging.info(f"Fetching history for #{ch_name} ({ch_id})")
                        if cursor:
                            logging.info(f"  Resuming from cursor {cursor}")
                        logging.info(f"  Sync direction: {state['sync_direction']}")

                    # Fetch one page of messages
                    params = {
                        "token": self.slack.token,
                        "channel": ch_id,
                        "limit": 200,
                    }
                    if cursor:
                        params["cursor"] = cursor

                    # Store current cursor before making the request
                    current_cursor = cursor

                    url = f"https://{self.slack.config.subdomain}.slack.com/api/conversations.history"
                    if self.slack.config.verbose:
                        logging.debug(f"GET {url}  cursor={cursor}")
                        self.slack._print_curl_command("GET", url, {}, params=params)

                    try:
                        r = self.slack.session.get(url, params=params)
                        r.raise_for_status()
                        j = r.json()
                        msgs = j.get("messages", [])
                    except Exception as e:
                        logging.error(f"Error fetching messages for channel {ch_name}: {e}")
                        # Restore cursor on error
                        state['cursor'] = current_cursor
                        continue

                    if msgs:
                        # Find the latest and oldest timestamps
                        latest_ts = None
                        oldest_ts = None
                        for m in msgs:
                            ts = m.get("ts")
                            if not ts:
                                logging.warning(f"Message without timestamp in channel {ch_name} ({ch_id})")
                                continue
                            try:
                                ts_float = float(ts)
                                if latest_ts is None or ts_float > latest_ts:
                                    latest_ts = ts_float
                                if oldest_ts is None or ts_float < oldest_ts:
                                    oldest_ts = ts_float
                            except (ValueError, TypeError) as e:
                                logging.warning(f"Invalid timestamp {ts} in channel {ch_name} ({ch_id}): {e}")
                                continue

                        if latest_ts is None or oldest_ts is None:
                            logging.warning(f"No valid timestamps in page {state['pages']} for channel {ch_name} ({ch_id})")
                            continue

                        # Update state timestamps
                        if state['latest_ts'] is None or latest_ts > state['latest_ts']:
                            state['latest_ts'] = latest_ts
                        if state['oldest_ts'] is None or oldest_ts < state['oldest_ts']:
                            state['oldest_ts'] = oldest_ts

                        state['pages'] += 1
                        try:
                            latest_time = datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S")
                            oldest_time = datetime.fromtimestamp(oldest_ts).strftime("%Y-%m-%d %H:%M:%S")
                            padded_name = f"#{ch_name}".ljust(max_channel_length + 1)  # +1 for the # symbol
                            print(f"{padded_name}: {len(msgs)} messages - {latest_time} to {oldest_time}")
                        except (ValueError, OSError) as e:
                            logging.warning(f"Error formatting timestamp: {e}")
                            padded_name = f"#{ch_name}".ljust(max_channel_length + 1)
                            print(f"{padded_name}: {len(msgs)} messages")

                    if self.slack.config.verbose:
                        logging.debug(f"  fetched {len(msgs)} msgs, next_cursor={cursor}")

                    # Process messages
                    for m in msgs:
                        ts = m.get("ts")
                        if not ts:
                            logging.warning(f"Skipping message without timestamp in channel {ch_name} ({ch_id})")
                            continue

                        try:
                            # Validate timestamp before storing
                            float(ts)  # Just to validate
                            self.db.store_compressed_data(
                                "messages", f"{ch_id}_{ts}", json.dumps(m, separators=(",", ":"))
                            )

                            # Use INSERT OR REPLACE to update existing messages
                            msg_cur.execute(
                                """
                                INSERT OR REPLACE INTO messages
                                (channel_id, ts, thread_ts, user_id, subtype, client_msg_id,
                                 edited_ts, edited_user, reply_count, reply_users_count,
                                 latest_reply, is_locked, has_files, has_blocks, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                            """,
                                (
                                    ch_id,
                                    ts,  # Store original string timestamp
                                    m.get("thread_ts"),
                                    m.get("user"),
                                    m.get("subtype"),
                                    m.get("client_msg_id"),
                                    m.get("edited", {}).get("ts"),
                                    m.get("edited", {}).get("user"),
                                    m.get("reply_count"),
                                    m.get("reply_users_count"),
                                    m.get("latest_reply"),
                                    bool(m.get("is_locked")),
                                    bool(m.get("files")),
                                    bool(m.get("blocks")),
                                ),
                            )

                            messages_processed += 1
                            state['messages'] += 1
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Error processing message with ts={ts} in channel {ch_name} ({ch_id}): {e}")
                            continue

                    self.db.conn.commit()

                    # Get next cursor and update sync state
                    next_cursor = j.get("response_metadata", {}).get("next_cursor")
                    
                    if state['sync_direction'] == 'forward':
                        if not next_cursor:
                            # End of forward sync, switch to backward
                            state['sync_direction'] = 'backward'
                            state['cursor'] = None
                            # Store the latest timestamp we've seen for backward sync
                            if state['latest_ts']:
                                sync_cur.execute(
                                    """
                                    INSERT OR REPLACE INTO sync_state
                                    (channel_id, last_sync_ts, last_sync_cursor, updated_at)
                                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                                """,
                                    (ch_id, str(state['latest_ts']), None),
                                )
                                self.db.conn.commit()
                        else:
                            # Continue forward sync
                            state['cursor'] = next_cursor
                            sync_cur.execute(
                                """
                                INSERT OR REPLACE INTO sync_state
                                (channel_id, last_sync_ts, last_sync_cursor, updated_at)
                                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                            """,
                                (ch_id, str(state['latest_ts']), next_cursor),
                            )
                            self.db.conn.commit()
                    else:  # backward sync
                        if not next_cursor:
                            # End of backward sync, check if we're fully synced
                            if state['oldest_ts'] and not state['is_fully_synced']:
                                sync_cur.execute(
                                    """
                                    UPDATE sync_state
                                    SET is_fully_synced = 1,
                                        last_sync_ts = ?,
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE channel_id = ?
                                """,
                                    (str(state['oldest_ts']), ch_id),
                                )
                                self.db.conn.commit()
                                state['is_fully_synced'] = True
                                channels_processed += 1
                                if not self.slack.config.verbose:
                                    print(f"  Completed backwards sync for #{ch_name}: {state['messages']} messages")
                            state['is_complete'] = True
                        else:
                            # Continue backward sync
                            state['cursor'] = next_cursor
                            sync_cur.execute(
                                """
                                INSERT OR REPLACE INTO sync_state
                                (channel_id, last_sync_ts, last_sync_cursor, updated_at)
                                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                            """,
                                (ch_id, str(state['oldest_ts']), next_cursor),
                            )
                            self.db.conn.commit()

                    # Update sync run progress
                    sync_cur.execute(
                        """
                        UPDATE sync_runs
                        SET channels_processed = ?, messages_processed = ?
                        WHERE id = ?
                    """,
                        (channels_processed, messages_processed, sync_run_id),
                    )
                    self.db.conn.commit()

                    if self.slack.config.verbose:
                        print(f"Sleeping for {self.slack.config.rate_limit} seconds")
                    time.sleep(self.slack.config.rate_limit)

            # Mark sync run as complete
            sync_cur.execute(
                """
                UPDATE sync_runs
                SET status = 'completed',
                    finished_at = CURRENT_TIMESTAMP,
                    channels_processed = ?,
                    messages_processed = ?
                WHERE id = ?
            """,
                (channels_processed, messages_processed, sync_run_id),
            )
            self.db.conn.commit()

            print(f"\nSync complete: {messages_processed} messages from {channels_processed} channels")
            return channels_processed, messages_processed

        except Exception as e:
            # Log error and mark sync run as failed
            logging.error(f"Sync failed: {str(e)}")
            sync_cur.execute(
                """
                UPDATE sync_runs
                SET status = 'failed',
                    error = ?,
                    finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (str(e), sync_run_id),
            )
            self.db.conn.commit()
            raise


class UserManager:
    """Manages user-related operations."""

    def __init__(self, db: DatabaseManager, slack: SlackClient):
        self.db = db
        self.slack = slack
        self._last_response = None

    def fetch_all_users(self) -> int:
        """Fetch all users from Slack, handling pagination."""
        cur = self.db.conn.cursor()
        marker = None
        users_processed = 0
        retry_count = 0
        max_retries = 3

        print("\nFetching users...")

        while True:
            try:
                # Prepare the request
                url = f"https://edgeapi.slack.com/cache/{self.slack.config.org_id}/users/list"
                params = {"_x_app_name": "client", "fp": "c7", "_x_num_retries": "0"}

                data = {
                    "token": self.slack.token,
                    "count": 1000,
                    "present_first": True,
                    "enterprise_token": self.slack.token,
                }

                if marker:
                    data["marker"] = marker

                headers = {
                    "accept": "*/*",
                    "accept-language": "en-US,en;q=0.9",
                    "cache-control": "no-cache",
                    "content-type": "text/plain;charset=UTF-8",
                    "origin": "https://app.slack.com",
                    "pragma": "no-cache",
                    "priority": "u=1, i",
                    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"macOS"',
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-site",
                }

                if self.slack.config.verbose:
                    logging.debug(f"POST {url}  marker={marker}")
                    # self.slack._print_curl_command("POST", url, headers, data=data, params=params)

                r = self.slack.session.post(
                    url, params=params, headers=headers, json=data
                )
                r.raise_for_status()
                self._last_response = r.json()

                # Process users
                users = self._last_response.get("results", [])
                if self.slack.config.verbose:
                    logging.debug(f"Got {len(users)} users")
                else:
                    print(f"Page: {len(users)} users")

                # Start transaction for bulk insert
                self.db.conn.execute("BEGIN TRANSACTION")

                for user in users:
                    user_id = user.get("id")
                    if not user_id:
                        continue

                    # Validate required fields
                    name = user.get("name", "").strip()
                    real_name = user.get("real_name", "").strip()
                    display_name = (
                        user.get("profile", {}).get("display_name", "").strip()
                    )

                    if not any([name, real_name, display_name]):
                        logging.warning(
                            f"Skipping user {user_id} - no valid name found"
                        )
                        continue

                    # Store user data
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO users
                        (id, name, real_name, display_name, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (user_id, name, real_name, display_name),
                    )

                    # Store compressed data
                    self.db.store_compressed_data(
                        "users", user_id, json.dumps(user, separators=(",", ":"))
                    )

                    users_processed += 1

                self.db.conn.commit()
                retry_count = 0  # Reset retry count on success

                # Check for more pages
                marker = self._get_next_marker()
                if not marker:
                    break

                if self.slack.config.verbose:
                    print(f"Sleeping for {self.slack.config.rate_limit} seconds")
                time.sleep(self.slack.config.rate_limit)

            except (requests.RequestException, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(
                        f"Failed to fetch users after {max_retries} retries: {str(e)}"
                    )
                    self.db.conn.rollback()
                    raise

                logging.warning(
                    f"Error fetching users (attempt {retry_count}/{max_retries}): {str(e)}"
                )
                self.db.conn.rollback()
                time.sleep(self.slack.config.rate_limit * 2)  # Exponential backoff

        print(f"\nProcessed {users_processed} users")
        return users_processed

    def _get_next_marker(self) -> Optional[str]:
        """Get the next marker from the response."""
        if not self._last_response:
            return None
        return self._last_response.get("next_marker")


class Exporter:
    """Handles data export operations."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def export_for_axolotl(self, output_dir: str) -> None:
        """Export database contents for Axolotl fine-tuning."""
        os.makedirs(output_dir, exist_ok=True)

        channels = self._get_all_channels()
        channel_files = []
        total_conversations = 0

        for channel_id, channel_name in channels:
            try:
                conversations = self._process_channel_conversations(
                    channel_id, channel_name
                )
                if not conversations:
                    continue

                output_path = self._write_channel_file(
                    channel_id, channel_name, conversations, output_dir
                )
                channel_files.append({"path": output_path, "type": "conversation"})
                total_conversations += len(conversations)

            except Exception as e:
                logging.error(f"Error processing channel {channel_name}: {str(e)}")
                continue

        self._write_metadata_files(
            output_dir, total_conversations, channels, channel_files
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Archive Slack channels via in-browser endpoints"
    )

    # Create a mutually exclusive group for the two modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "subdomain",
        nargs="?",
        help="Slack workspace subdomain (e.g. datavant.enterprise)",
    )
    mode_group.add_argument(
        "--dump-axolotl",
        help="Export database contents for Axolotl fine-tuning to the specified directory",
    )

    # Add all other arguments as optional
    parser.add_argument(
        "--org", help="Which specific Slack org the cookie is signed into"
    )
    parser.add_argument(
        "--x-version-timestamp", help="X-Version-Timestamp from the Slack homepage"
    )
    parser.add_argument(
        "--team", help="Which specific Slack team the cookie is signed into"
    )
    parser.add_argument("--cookie", help="Your full Slack session cookie string")
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between HTTP requests",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--client-req-id", help="Client request ID for API calls")
    parser.add_argument("--browse-session-id", help="Browse session ID for API calls")

    args = parser.parse_args()

    # For archive mode, validate required arguments
    if not args.dump_axolotl and not all(
        [args.subdomain, args.org, args.x_version_timestamp, args.team, args.cookie]
    ):
        parser.error(
            "When archiving (not using --dump-axolotl), the following arguments are required: subdomain, --org, --x-version-timestamp, --team, --cookie"
        )

    return args


def create_slack_config(args: argparse.Namespace) -> SlackConfig:
    """Create SlackConfig from command line arguments."""
    return SlackConfig(
        subdomain=args.subdomain,
        org_id=args.org,
        team_id=args.team,
        x_version_timestamp=args.x_version_timestamp,
        cookie=args.cookie,
        rate_limit=args.rate_limit,
        verbose=args.verbose,
        client_req_id=args.client_req_id,
        browse_session_id=args.browse_session_id,
    )


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    db = DatabaseManager()
    db.connect()

    try:
        if args.dump_axolotl:
            exporter = Exporter(db)
            exporter.export_for_axolotl(args.dump_axolotl)
            return

        config = create_slack_config(args)
        slack = SlackClient(config)
        slack.initialize()

        user_manager = UserManager(db, slack)
        channel_manager = ChannelManager(db, slack)
        message_manager = MessageManager(db, slack)

        #user_manager.fetch_all_users()
        #channel_manager.fetch_all_channels()
        message_manager.sync_all_channels()

        logging.info("Done! Archive stored in %s", DB_PATH)

    finally:
        db.close()


if __name__ == "__main__":
    main()
