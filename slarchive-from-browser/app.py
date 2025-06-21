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

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self) -> None:
        """Establish database connection and setup schema."""
        self.conn = sqlite3.connect(self.db_path)
        self._setup_schema()

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

        # Create all tables
        tables = [
            self._get_channels_schema(),
            self._get_messages_schema(),
            self._get_users_schema(),
            self._get_compressed_data_schema(),
        ]

        for schema in tables:
            cur.execute(schema)

        self.conn.commit()

    def _get_channels_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS channels (
            id TEXT PRIMARY KEY,
            name TEXT,
            ever_fully_synced BOOLEAN DEFAULT FALSE,
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (channel_id, ts)
        );
        """

    def _get_users_schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            real_name TEXT,
            display_name TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

    def list_all_channels(self) -> List[Dict]:
        """List all channels from Slack."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT id, name FROM channels")
        return cur.fetchall()

    def fetch_all_channels(self) -> int:
        """Fetch all channels from Slack."""
        cur = self.db.conn.cursor()
        page = 1
        per_page = 100
        channels_processed = 0

        channel_types = [
            "external_shared",
            "exclude_archived",
            "private_exclude",
            "archived",
        ]
        items = []

        for channel_type in channel_types:
            print(f"\nFetching channels for {channel_type}...")

            while True:
                channels = self._fetch_channel_page(page, per_page, channel_type)
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

    def _fetch_channel_page(self, page: int, per_page: int, channel_type: str) -> List[Dict]:
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
        form_data.append(
            'Content-Disposition: form-data; name="extra_message_data"'
        )
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
        form_data.append(
            'Content-Disposition: form-data; name="query_rewrite_disabled"'
        )
        form_data.append("")
        form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append(
            'Content-Disposition: form-data; name="include_files_shares"'
        )
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
        form_data.append(
            'Content-Disposition: form-data; name="max_filter_suggestions"'
        )
        form_data.append("")
        form_data.append("1000")

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
        form_data.append(channel_type)

        form_data.append(f"--{boundary}")
        form_data.append(
            'Content-Disposition: form-data; name="exclude_my_channels"'
        )
        form_data.append("")
        form_data.append("0")

        # form_data.append(f"--{boundary}")
        # form_data.append('Content-Disposition: form-data; name="search_only_team"')
        # form_data.append("")
        # form_data.append(self.slack.config.team_id)

        form_data.append(f"--{boundary}")
        form_data.append(
            'Content-Disposition: form-data; name="search_recently_left_channels"'
        )
        form_data.append("")
        form_data.append("false")

        form_data.append(f"--{boundary}")
        form_data.append(
            'Content-Disposition: form-data; name="search_recently_joined_channels"'
        )
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
            self.slack._print_curl_command(
                "POST", url, headers, data=data, params=params
            )

        r = self.slack.session.post(url, headers=headers, data=data, params=params)
        r.raise_for_status()
        j = r.json()

        # adjust this if your workspace returns under a different key
        return j.get("items", [])


class MessageProcessor:
    """Handles processing and storing individual messages."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.msg_cur = db.conn.cursor()

    def process_messages(
        self, channel_id: str, messages: List[Dict]
    ) -> Tuple[int, Optional[float], Optional[float]]:
        """Process a batch of messages and return count and timestamp range."""
        messages_processed = 0
        latest_ts = None
        oldest_ts = None

        for m in messages:
            ts = m.get("ts")
            if not ts:
                continue

            try:
                # Validate timestamp
                ts_float = float(ts)
                if latest_ts is None or ts_float > latest_ts:
                    latest_ts = ts_float
                if oldest_ts is None or ts_float < oldest_ts:
                    oldest_ts = ts_float

                # Store message
                self._store_message(channel_id, m)
                messages_processed += 1

            except (ValueError, TypeError) as e:
                logging.warning(f"Error processing message with ts={ts}: {e}")
                continue

        return messages_processed, latest_ts, oldest_ts

    def _store_message(self, channel_id: str, message: Dict) -> None:
        """Store a single message in the database."""
        ts = message.get("ts")

        # Store compressed data
        self.db.store_compressed_data(
            "messages", f"{channel_id}_{ts}", json.dumps(message, separators=(",", ":"))
        )

        # Store message metadata
        self.msg_cur.execute(
            """
            INSERT OR REPLACE INTO messages
            (channel_id, ts, thread_ts, user_id, subtype, client_msg_id,
             edited_ts, edited_user, reply_count, reply_users_count,
             latest_reply, is_locked, has_files, has_blocks, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (
                channel_id,
                ts,
                message.get("thread_ts"),
                message.get("user"),
                message.get("subtype"),
                message.get("client_msg_id"),
                message.get("edited", {}).get("ts"),
                message.get("edited", {}).get("user"),
                message.get("reply_count"),
                message.get("reply_users_count"),
                message.get("latest_reply"),
                bool(message.get("is_locked")),
                bool(message.get("files")),
                bool(message.get("blocks")),
            ),
        )


class MessageManager:
    """Manages message-related operations."""

    def __init__(
        self, db: DatabaseManager, channel_manager: ChannelManager, slack: SlackClient
    ):
        self.db = db
        self.slack = slack
        self.channel_manager = channel_manager
        self.message_processor = MessageProcessor(db)

    def sync_all_channels(self) -> Tuple[int, int]:
        """Sync messages from all channels, one at a time."""
        channels_processed = 0
        messages_processed = 0

        try:
            # Get all channels
            channels = self.channel_manager.list_all_channels()
            total_channels = len(channels)

            print(f"\nStarting sync of {total_channels} channels...")

            # Process each channel
            for ch_id, ch_name in channels:
                channel_msgs, channel_success = self.sync_channel(ch_id, ch_name)
                if channel_success:
                    channels_processed += 1
                messages_processed += channel_msgs

            print(
                f"\nSync complete: {messages_processed} messages from {channels_processed} channels"
            )
            return channels_processed, messages_processed

        except Exception as e:
            logging.error(f"Sync failed: {str(e)}")
            raise

    def sync_channel(self, channel_id: str, channel_name: str) -> Tuple[int, bool]:
        """
        If the channel has never been fully synced, read pages of messages until
        we reach the first ever posted in the channel and set ever_fully_synced to true.
        If we get interrupted along the way, continue from the oldest (lowest) ts in the channel history.
        If the channel has ever been fully synced, we'll read from the newest
        and stop when we encounter an existing timestamp.
        """

        # Get the newest timestamp we've seen for this channel
        cur = self.db.conn.cursor()
        cur.execute("SELECT max(ts) FROM messages WHERE channel_id = ?", (channel_id,))
        row = cur.fetchone()
        latest_synced_ts = row[0] if row and row[0] is not None else None

        # Get the oldest timestamp we've seen for this channel
        cur = self.db.conn.cursor()
        cur.execute("SELECT min(ts) FROM messages WHERE channel_id = ?", (channel_id,))
        row = cur.fetchone()
        oldest_synced_ts = row[0] if row and row[0] is not None else None

        messages_processed = 0
        batch_size = 0
        has_more = True
        batch_oldest_ts = None

        cur.execute("SELECT ever_fully_synced FROM channels WHERE id = ?", (channel_id,))
        row = cur.fetchone()
        ever_fully_synced = row[0] if row and row[0] is not None else False

        if ever_fully_synced:
            latest_ts = None
        else:
            latest_ts = oldest_synced_ts

        while has_more:
            # Fetch messages, starting from latest_cursor or beginning
            messages, has_more, latest_ts = self._fetch_messages(
                channel_id, latest_ts
            )

            # If we got no messages, we're done with this channel
            fully_synced = not has_more


            # Check if we've reached previously synced messages
            if latest_synced_ts is not None:
                # Find first message with timestamp <= oldest_ts_synced
                for i, msg in enumerate(messages):
                    if float(msg.get("ts", 0)) <= float(latest_synced_ts):
                        # We've reached previously synced messages, only process newer ones
                        messages = messages[:i]
                        has_more = False  # Stop fetching more
                        break

            # Process this batch of messages
            if messages:
                batch_size, latest_ts, batch_oldest_ts = (
                    self.message_processor.process_messages(channel_id, messages)
                )
                messages_processed += batch_size

                
                oldest_readable = datetime.fromtimestamp(batch_oldest_ts).strftime('%Y-%m-%d %H:%M:%S')
                latest_readable = datetime.fromtimestamp(latest_ts).strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"#{channel_name}: Processed {batch_size} messages (total: {messages_processed}) - {oldest_readable} to {latest_readable}"
                )

            if fully_synced:
                self.db.conn.execute("UPDATE channels SET ever_fully_synced = ? WHERE id = ?", (fully_synced, channel_id))
                self.db.conn.commit()
                print(f"#{channel_name} fully synced")

            # Update cursor for next batch
            latest_ts = batch_oldest_ts

            # Sleep to respect rate limits
            time.sleep(self.slack.config.rate_limit)

        return messages_processed, messages_processed > 0

    def _fetch_messages(
        self, channel_id: str, latest_ts: Optional[str]
    ) -> Tuple[List[Dict], bool, Optional[str]]:
        """Fetch a batch of messages from a channel.

        Returns:
            Tuple of (messages, has_more, next_cursor)
        """
        url = (
            f"https://{self.slack.config.subdomain}.slack.com/api/conversations.history"
        )

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
        form_data.append('Content-Disposition: form-data; name="channel"')
        form_data.append("")
        form_data.append(str(channel_id))

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="no_user_profile"')
        form_data.append("")
        form_data.append("true")

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="limit"')
        form_data.append("")
        form_data.append("200")

        # Add latest parameter if we have one
        if latest_ts is not None:
            form_data.append(f"--{boundary}")
            form_data.append('Content-Disposition: form-data; name="latest"')
            form_data.append("")
            form_data.append(str(latest_ts))

        form_data.append(f"--{boundary}")
        form_data.append('Content-Disposition: form-data; name="_x_reason"')
        form_data.append("")
        form_data.append("message-pane/requestHistory")

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
            logging.debug(f"POST {url}  latest_ts={latest_ts}")
            self.slack._print_curl_command(
                "POST", url, headers, data=data, params=params
            )

        try:
            r = self.slack.session.post(url, headers=headers, data=data, params=params)
            r.raise_for_status()
            j = r.json()

            messages = j.get("messages", [])
            has_more = j.get("has_more", False)
            if messages:
                next_latest = min([float(m.get("ts", 0)) for m in messages])
            else:
                next_latest = None

            return messages, has_more, next_latest

        except Exception as e:
            logging.error(f"Error fetching messages for channel {channel_id}: {e}")
            return [], False, None


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
    """Handles data export operations for LLM fine-tuning."""

    MAX_CONTEXT_MESSAGES = 10  # Maximum number of messages to include in context
    MAX_TOKENS_PER_CONVERSATION = 2048  # Approximate max tokens per conversation

    def __init__(self, db: DatabaseManager):
        self.db = db

    def _get_all_channels(self) -> List[Tuple[str, str]]:
        """Get all channels from the database."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT id, name FROM channels ORDER BY name")
        return cur.fetchall()

    def _get_thread_messages(self, channel_id: str, thread_ts: str) -> List[Dict]:
        """Get all messages in a thread."""
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT m.ts, m.user_id, m.thread_ts, m.subtype, m.client_msg_id,
                   m.edited_ts, m.edited_user, m.reply_count, m.reply_users_count,
                   m.latest_reply, m.is_locked, m.has_files, m.has_blocks
            FROM messages m
            WHERE m.channel_id = ? AND m.thread_ts = ?
            ORDER BY m.ts
        """,
            (channel_id, thread_ts),
        )
        thread_messages = []
        for msg in cur.fetchall():
            ts = msg[0]
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if raw_data:
                thread_messages.append(json.loads(raw_data))
        return thread_messages

    def _estimate_tokens(self, text: str) -> int:
        """Roughly estimate the number of tokens in a text."""
        # Very rough approximation: 4 characters per token
        return len(text) // 4

    def _format_message(
        self, msg: Dict, user_name: str, include_reactions: bool = True
    ) -> str:
        """Format a single message with user and content."""
        text = msg.get("text", "").strip()

        # Add any file information
        files = msg.get("files", [])
        if files:
            file_info = [f"[Shared file: {f.get('name', 'unnamed')}]" for f in files]
            text += "\n" + "\n".join(file_info)

        # Add reactions if present and requested
        if include_reactions and msg.get("reactions"):
            reaction_text = []
            for reaction in msg["reactions"]:
                count = reaction.get("count", 0)
                users = reaction.get("users", [])
                if count and users:
                    reaction_text.append(f":{reaction['name']}: × {count}")
            if reaction_text:
                text += "\n[Reactions: " + " ".join(reaction_text) + "]"

        # Format edited information
        if msg.get("edited"):
            text += (
                f"\n[edited by {self.db.get_user_display_name(msg['edited']['user'])}]"
            )

        return f"{user_name}: {text}"

    def _process_channel_conversations(
        self, channel_id: str, channel_name: str
    ) -> List[Dict]:
        """Process channel messages into coherent conversation chunks for fine-tuning."""
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT m.ts, m.user_id, m.thread_ts, m.subtype
            FROM messages m
            WHERE m.channel_id = ?
            ORDER BY m.ts
        """,
            (channel_id,),
        )

        conversations = []
        current_conversation = {
            "channel": channel_name,
            "messages": [],
            "token_count": 0,
        }

        last_ts = None
        time_gap_threshold = 3600 * 24  # 1 day in seconds

        for msg_row in cur.fetchall():
            ts, user_id, thread_ts, subtype = msg_row

            # Skip system messages
            if subtype in ["channel_join", "channel_leave", "bot_message"]:
                continue

            # Get the full message data
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if not raw_data:
                continue

            msg_data = json.loads(raw_data)
            user_name = self.db.get_user_display_name(user_id)

            # Format the message
            formatted_msg = self._format_message(msg_data, user_name)
            msg_tokens = self._estimate_tokens(formatted_msg)

            # Check if this is a thread starter
            if not thread_ts and msg_data.get("reply_count", 0) > 0:
                # Get thread messages
                thread_msgs = self._get_thread_messages(channel_id, ts)
                thread_text = "\n".join(
                    [
                        self._format_message(
                            tm,
                            self.db.get_user_display_name(tm.get("user")),
                            include_reactions=False,
                        )
                        for tm in thread_msgs
                    ]
                )
                formatted_msg += f"\n[Thread responses:\n{thread_text}\n]"
                msg_tokens = self._estimate_tokens(formatted_msg)

            # Start a new conversation if:
            # 1. Adding this message would exceed token limit
            # 2. Time gap is too large
            # 3. Current conversation already has max messages
            if (
                current_conversation["token_count"] + msg_tokens
                > self.MAX_TOKENS_PER_CONVERSATION
                or (last_ts and float(ts) - float(last_ts) > time_gap_threshold)
                or len(current_conversation["messages"]) >= self.MAX_CONTEXT_MESSAGES
            ):

                if current_conversation["messages"]:
                    conversations.append(current_conversation)
                current_conversation = {
                    "channel": channel_name,
                    "messages": [],
                    "token_count": 0,
                }

            # Add message to current conversation
            current_conversation["messages"].append(
                {"timestamp": ts, "user": user_name, "content": formatted_msg}
            )
            current_conversation["token_count"] += msg_tokens
            last_ts = ts

        # Add the last conversation if it has messages
        if current_conversation["messages"]:
            conversations.append(current_conversation)

        return conversations

    def _write_channel_file(
        self,
        channel_id: str,
        channel_name: str,
        conversations: List[Dict],
        output_dir: str,
    ) -> str:
        """Write channel conversations to a file in a format suitable for fine-tuning."""
        output_path = os.path.join(output_dir, f"{channel_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                # Format for axolotl fine-tuning
                formatted_conv = {
                    "conversations": [
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant that understands Slack conversations and can answer questions about them.",
                        },
                        {
                            "role": "user",
                            "content": f"Channel: #{conv['channel']}\n\n"
                            + "\n".join(msg["content"] for msg in conv["messages"]),
                        },
                        {
                            "role": "assistant",
                            "content": "I understand the conversation and can answer questions about its content, participants, and key points discussed.",
                        },
                    ]
                }
                f.write(json.dumps(formatted_conv, ensure_ascii=False) + "\n")
        return output_path

    def _write_metadata_files(
        self,
        output_dir: str,
        total_conversations: int,
        channels: List[Tuple[str, str]],
        channel_files: List[Dict],
    ) -> None:
        """Write metadata files for the export."""
        summary = {
            "total_conversations": total_conversations,
            "total_channels": len(channels),
            "channels": [{"id": ch_id, "name": ch_name} for ch_id, ch_name in channels],
            "files": channel_files,
            "exported_at": datetime.now().isoformat(),
            "format_info": {
                "max_context_messages": self.MAX_CONTEXT_MESSAGES,
                "max_tokens_per_conversation": self.MAX_TOKENS_PER_CONVERSATION,
                "format": "instruction-input-output pairs with metadata",
                "instruction_template": "Given the following Slack conversation...",
                "usage_notes": [
                    "Each conversation is limited to prevent context window overflow",
                    "Threads are included with their parent messages",
                    "System messages are filtered out",
                    "Reactions and edits are preserved as contextual information",
                    "File shares are noted but content is not included",
                ],
            },
        }

        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Write a README with usage instructions
        readme_content = f"""# Slack Conversation Training Data

This directory contains Slack conversations exported for LLM fine-tuning.

## Format

Each .jsonl file contains conversations from a single channel, formatted as instruction-input-output pairs:

```json
{{
    "instruction": "Given the following Slack conversation...",
    "input": "Channel: #channel-name\\nuser1: message1\\nuser2: message2...",
    "output": "I understand the conversation...",
    "metadata": {{...}}
}}
```

## Usage Notes

- Maximum context size: {self.MAX_TOKENS_PER_CONVERSATION} tokens per conversation
- Maximum messages per context: {self.MAX_CONTEXT_MESSAGES}
- Conversations are split based on:
  - Token limit
  - Time gaps (>1 hour)
  - Message count
- Thread responses are included with their parent messages
- Reactions and edits are preserved as contextual information

## Statistics

- Total conversations: {total_conversations:,}
- Total channels: {len(channels):,}
- Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

See summary.json for detailed statistics and channel information.
"""
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

    def export_for_axolotl(self, output_dir: str) -> None:
        """Export database contents in a format suitable for LLM fine-tuning."""
        os.makedirs(output_dir, exist_ok=True)

        channels = self._get_all_channels()
        channel_files = []
        total_conversations = 0

        print(f"\nExporting {len(channels)} channels to {output_dir}...")

        for channel_id, channel_name in channels:
            try:
                print(f"Processing channel #{channel_name}...")
                conversations = self._process_channel_conversations(
                    channel_id, channel_name
                )
                if not conversations:
                    print(f"  No conversations found in #{channel_name}")
                    continue

                output_path = self._write_channel_file(
                    channel_id, channel_name, conversations, output_dir
                )
                channel_files.append(
                    {
                        "path": output_path,
                        "type": "conversation",
                        "conversation_count": len(conversations),
                    }
                )
                total_conversations += len(conversations)
                print(f"  Exported {len(conversations)} conversations")

            except Exception as e:
                logging.error(f"Error processing channel {channel_name}: {str(e)}")
                continue

        self._write_metadata_files(
            output_dir, total_conversations, channels, channel_files
        )
        print(
            f"\nExport complete: {total_conversations:,} conversations from {len(channels)} channels"
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
        "--axolotl",
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
    if not args.axolotl and not all(
        [args.subdomain, args.org, args.x_version_timestamp, args.team, args.cookie]
    ):
        parser.error(
            "When archiving (not using --axolotl), the following arguments are required: subdomain, --org, --x-version-timestamp, --team, --cookie"
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
        if args.axolotl:
            exporter = Exporter(db)
            exporter.export_for_axolotl(args.axolotl)
            return

        config = create_slack_config(args)
        slack = SlackClient(config)
        slack.initialize()

        user_manager = UserManager(db, slack)
        channel_manager = ChannelManager(db, slack)
        message_manager = MessageManager(db, channel_manager, slack)

        print("\n=== Starting Slack Archive Sync ===\n")

        # Step 1: Fetch users
        user_count = user_manager.fetch_all_users()

        # Step 2: Fetch channels
        channel_count = channel_manager.fetch_all_channels()

        # Step 3: Sync messages from all channels
        messages_processed = message_manager.sync_all_channels()

        print("\n=== Sync Complete ===")
        print(f"Users: {user_count}")
        print(f"Channels: {channel_count}")
        print(f"Messages: {messages_processed}")
        print(f"Archive stored in {DB_PATH}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
#
