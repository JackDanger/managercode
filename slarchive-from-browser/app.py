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
        
        # Performance optimizations
        self.conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
        self.conn.execute("PRAGMA synchronous = NORMAL")  # Sync less often (1=NORMAL, 2=FULL, 0=OFF)
        self.conn.execute("PRAGMA cache_size = 10000")  # Increase cache size (in pages)
        self.conn.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory
        
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

    def _fetch_channel_page(
        self, page: int, per_page: int, channel_type: str
    ) -> List[Dict]:
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
        form_data.append(
            'Content-Disposition: form-data; name="query_rewrite_disabled"'
        )
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
        form_data.append('Content-Disposition: form-data; name="exclude_my_channels"')
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
        
        # Start a single transaction for the entire batch
        self.db.conn.execute("BEGIN TRANSACTION")
        
        try:
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
            
            # Commit the batch of inserts
            self.db.conn.commit()
            
        except Exception as e:
            # Rollback on error
            self.db.conn.rollback()
            logging.error(f"Error processing message batch: {e}")
            raise

        return messages_processed, latest_ts, oldest_ts

    def _store_message(self, channel_id: str, message: Dict) -> None:
        """Store a single message in the database."""
        ts = message.get("ts")

        data = json.dumps(message, separators=(",", ":"))
        # Compress the data
        compressed = self.db.compress_data(data)
        original_size = len(data.encode("utf-8"))
        compressed_size = len(compressed)

        # Store compressed data
        comp_cur = self.db.conn.cursor()
        comp_cur.execute(
            """
            INSERT OR REPLACE INTO compressed_data
            (table_name, record_id, compressed_data, original_size, compressed_size, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            ("messages", f"{channel_id}_{ts}", compressed, original_size, compressed_size),
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
        cur.execute("SELECT min(ts) FROM messages WHERE channel_id = ?", (channel_id,))
        row = cur.fetchone()
        oldest_synced_ts = row[0] if row and row[0] is not None else None

        messages_processed = 0
        batch_size = 0
        has_more = True
        batch_oldest_ts = None

        cur.execute(
            "SELECT ever_fully_synced FROM channels WHERE id = ?", (channel_id,)
        )
        row = cur.fetchone()
        ever_fully_synced = row[0] if row and row[0] is not None else False

        if ever_fully_synced:
            latest_ts = None
        else:
            latest_ts = oldest_synced_ts

        while has_more:
            # Fetch messages, starting from latest_cursor or beginning
            messages, has_more, latest_ts, http_timing = self._fetch_messages(channel_id, latest_ts)

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
                start_time = time.time()
                batch_size, latest_ts, batch_oldest_ts = (
                    self.message_processor.process_messages(channel_id, messages)
                )
                messages_processed += batch_size

                oldest_readable = datetime.fromtimestamp(batch_oldest_ts).strftime(
                    "%Y-%m-%d"
                )
                latest_readable = datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d")
                message_count = str(messages_processed).rjust(5)
                http_timing_str = f"{float(http_timing):.2f}s"
                db_timing_str = f"{float(start_time - time.time()):.2f}s"
                print(
                    f"#{channel_name.ljust(70)}: Processed {str(batch_size).ljust(3)} messages (total: {message_count}) - {latest_readable} to {oldest_readable} in http:{http_timing_str} db:{db_timing_str}"
                )

            if fully_synced:
                self.db.conn.execute(
                    "UPDATE channels SET ever_fully_synced = ? WHERE id = ?",
                    (fully_synced, channel_id),
                )
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

        try:
            r = self.slack.session.post(url, headers=headers, data=data, params=params)
            http_timing = r.elapsed.total_seconds()
            r.raise_for_status()
            j = r.json()

            messages = j.get("messages", [])
            has_more = j.get("has_more", False)
            if messages:
                next_latest = min([float(m.get("ts", 0)) for m in messages])
            else:
                next_latest = None

            if self.slack.config.verbose:
                logging.debug(f"POST {url}  latest_ts={latest_ts}")

            return messages, has_more, next_latest, http_timing

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

                r = self.slack.session.post(
                    url, params=params, headers=headers, json=data
                )
                r.raise_for_status()
                http_timing = r.elapsed.total_seconds()
                self._last_response = r.json()

                # Process users
                users = self._last_response.get("results", [])
                print(f"Page: {len(users)} users")

                start_time = time.time()
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
                sql_timing = time.time() - start_time
                if self.slack.config.verbose:
                    print(f"Timing: http={http_timing}, sql={sql_timing}")
                retry_count = 0  # Reset retry count on success

                # Check for more pages
                marker = self._get_next_marker()
                if not marker:
                    break

                if self.slack.config.verbose:
                    print(f"Sleeping for {self.slack.config.rate_limit} seconds")
                time.sleep(self.slack.config.rate_limit * 2)  # Exponential backoff

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

    # Conservative defaults for basic export
    MAX_CONTEXT_MESSAGES = 10
    MAX_TOKENS_PER_CONVERSATION = 2048
    
    # Enhanced settings for comprehensive knowledge capture
    ENHANCED_MAX_TOKENS = 8192  # For long-form content
    ENHANCED_MAX_MESSAGES = 50  # More context for complex discussions
    KNOWLEDGE_EXTRACTION_TOKENS = 16384  # For deep technical discussions
    
    # Thresholds for identifying valuable content
    MIN_MESSAGE_LENGTH = 100  # Characters - filter out short messages
    THREAD_IMPORTANCE_THRESHOLD = 3  # Minimum replies to consider thread important
    TECHNICAL_KEYWORDS = [
        'architecture', 'implementation', 'design', 'algorithm', 'process',
        'workflow', 'documentation', 'specification', 'requirements', 'analysis',
        'solution', 'approach', 'methodology', 'framework', 'system', 'technical',
        'engineering', 'development', 'deployment', 'configuration', 'integration',
        'troubleshooting', 'debugging', 'optimization', 'performance', 'security',
        'compliance', 'audit', 'review', 'assessment', 'evaluation', 'research'
    ]

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
        """More accurate token estimation using tiktoken-style approximation."""
        # Better approximation: ~3.3 characters per token for English text
        return int(len(text) / 3.3)

    def _calculate_content_importance(self, msg_data: Dict, thread_msgs: List[Dict] = None) -> float:
        """Calculate importance score for content based on multiple factors."""
        score = 0.0
        text = msg_data.get("text", "")
        
        # Length factor (longer messages often contain more knowledge)
        if len(text) > 500:
            score += 3.0
        elif len(text) > 200:
            score += 2.0
        elif len(text) > 100:
            score += 1.0
        
        # Technical keyword density
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.TECHNICAL_KEYWORDS if keyword in text_lower)
        score += keyword_count * 0.5
        
        # Thread engagement (replies indicate valuable discussion)
        reply_count = msg_data.get("reply_count", 0)
        if reply_count > 10:
            score += 4.0
        elif reply_count > 5:
            score += 2.0
        elif reply_count > 2:
            score += 1.0
        
        # File attachments (often contain important documentation)
        if msg_data.get("files"):
            score += 2.0
        
        # Code blocks or formatted content
        if "```" in text or "`" in text:
            score += 1.5
        
        # URLs (often reference important resources)
        if "http" in text:
            score += 0.5
        
        # Reactions (community validation of importance)
        reactions = msg_data.get("reactions", [])
        if reactions:
            total_reactions = sum(r.get("count", 0) for r in reactions)
            score += min(total_reactions * 0.1, 2.0)  # Cap at 2.0
        
        return score

    def _format_message(
        self, msg: Dict, user_name: str, include_reactions: bool = True, include_metadata: bool = False
    ) -> str:
        """Enhanced message formatting with optional metadata."""
        text = msg.get("text", "").strip()
        
        # Add timestamp if requested
        if include_metadata:
            ts = msg.get("ts", "")
            if ts:
                dt = datetime.fromtimestamp(float(ts))
                text = f"[{dt.strftime('%Y-%m-%d %H:%M')}] {text}"

        # Add any file information with more detail
        files = msg.get("files", [])
        if files:
            file_info = []
            for f in files:
                name = f.get("name", 'unnamed')
                filetype = f.get("filetype", 'unknown')
                size = f.get("size", 0)
                if size > 0:
                    size_kb = size // 1024
                    file_info.append(f"[File: {name} ({filetype}, {size_kb}KB)]")
                else:
                    file_info.append(f"[File: {name} ({filetype})]")
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
            edited_user = self.db.get_user_display_name(msg['edited']['user'])
            text += f"\n[edited by {edited_user}]"

        return f"{user_name}: {text}"

    def _create_knowledge_extraction_prompt(self, conversation: Dict, context_type: str) -> Dict:
        """Create sophisticated prompts for knowledge extraction."""
        content = "\n".join(msg["content"] for msg in conversation["messages"])
        
        prompts = {
            "concept_explanation": {
                "system": "You are an expert at extracting and explaining technical concepts from workplace discussions. Your role is to identify key concepts, processes, and knowledge shared in conversations and explain them clearly and comprehensively.",
                "user": f"Analyze this Slack conversation from #{conversation['channel']} and extract any technical concepts, processes, or organizational knowledge that was explained or discussed:\n\n{content}\n\nProvide a comprehensive explanation of the key concepts, including context about how they're used in this organization.",
                "assistant": "I'll analyze this conversation and extract the key technical concepts and organizational knowledge, providing clear explanations with proper context."
            },
            "problem_solution": {
                "system": "You are an expert at identifying problems and their solutions from workplace discussions. You excel at capturing troubleshooting knowledge, workarounds, and solution methodologies.",
                "user": f"From this Slack conversation in #{conversation['channel']}, identify any problems that were discussed and their solutions:\n\n{content}\n\nDocument the problem-solving approach, any troubleshooting steps, and the final resolution with enough detail that someone could apply this knowledge to similar situations.",
                "assistant": "I'll identify the problems discussed and document the solutions and troubleshooting approaches in detail for future reference."
            },
            "process_documentation": {
                "system": "You are an expert at documenting organizational processes and workflows from informal discussions. You can identify implicit processes and make them explicit.",
                "user": f"Extract any organizational processes, workflows, or procedures mentioned in this Slack conversation from #{conversation['channel']}:\n\n{content}\n\nDocument these processes clearly, including steps, responsibilities, tools used, and any important considerations or edge cases mentioned.",
                "assistant": "I'll extract and document the organizational processes and workflows discussed, making implicit knowledge explicit and actionable."
            },
            "decision_rationale": {
                "system": "You are an expert at capturing decision-making rationale and the reasoning behind organizational choices. You understand the importance of preserving the 'why' behind decisions.",
                "user": f"Identify any decisions made or decision-making discussions in this Slack conversation from #{conversation['channel']}:\n\n{content}\n\nCapture the rationale, alternatives considered, trade-offs discussed, and the reasoning behind the final decisions.",
                "assistant": "I'll identify the decisions made and capture the complete rationale and reasoning process for future reference."
            }
        }
        
        return prompts.get(context_type, prompts["concept_explanation"])

    def _process_channel_conversations_enhanced(
        self, channel_id: str, channel_name: str, strategy: str = "comprehensive"
    ) -> List[Dict]:
        """Enhanced conversation processing with multiple strategies."""
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
        
        if strategy == "knowledge_focused":
            # Focus on high-value knowledge content
            return self._process_knowledge_focused(channel_id, channel_name, cur.fetchall())
        elif strategy == "thread_complete":
            # Ensure complete thread capture
            return self._process_thread_complete(channel_id, channel_name, cur.fetchall())
        elif strategy == "temporal_context":
            # Maintain temporal relationships
            return self._process_temporal_context(channel_id, channel_name, cur.fetchall())
        else:
            # Comprehensive approach combining all strategies
            return self._process_comprehensive(channel_id, channel_name, cur.fetchall())

    def _process_knowledge_focused(self, channel_id: str, channel_name: str, msg_rows: List) -> List[Dict]:
        """Process focusing on high-knowledge-value content."""
        conversations = []
        high_value_messages = []
        
        for msg_row in msg_rows:
            ts, user_id, thread_ts, subtype = msg_row
            
            if subtype in ["channel_join", "channel_leave", "bot_message"]:
                continue
                
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if not raw_data:
                continue
                
            msg_data = json.loads(raw_data)
            
            # Calculate importance score
            thread_msgs = []
            if not thread_ts and msg_data.get("reply_count", 0) > 0:
                thread_msgs = self._get_thread_messages(channel_id, ts)
            
            importance = self._calculate_content_importance(msg_data, thread_msgs)
            
            # Only include high-value content
            if importance >= 2.0:  # Threshold for knowledge-worthy content
                user_name = self.db.get_user_display_name(user_id)
                formatted_msg = self._format_message(msg_data, user_name, include_metadata=True)
                
                # Include thread content for complete context
                if thread_msgs:
                    thread_text = "\n".join([
                        self._format_message(
                            tm, 
                            self.db.get_user_display_name(tm.get("user")),
                            include_reactions=False,
                            include_metadata=True
                        ) for tm in thread_msgs
                    ])
                    formatted_msg += f"\n[Thread responses:\n{thread_text}\n]"
                
                high_value_messages.append({
                    "timestamp": ts,
                    "user": user_name,
                    "content": formatted_msg,
                    "importance": importance,
                    "tokens": self._estimate_tokens(formatted_msg)
                })
        
        # Group high-value messages into conversations with larger context windows
        current_conversation = {
            "channel": channel_name,
            "messages": [],
            "token_count": 0,
            "importance_score": 0.0
        }
        
        for msg in high_value_messages:
            if (current_conversation["token_count"] + msg["tokens"] > self.KNOWLEDGE_EXTRACTION_TOKENS or
                len(current_conversation["messages"]) >= self.ENHANCED_MAX_MESSAGES):
                
                if current_conversation["messages"]:
                    conversations.append(current_conversation)
                current_conversation = {
                    "channel": channel_name,
                    "messages": [],
                    "token_count": 0,
                    "importance_score": 0.0
                }
            
            current_conversation["messages"].append(msg)
            current_conversation["token_count"] += msg["tokens"]
            current_conversation["importance_score"] += msg["importance"]
        
        if current_conversation["messages"]:
            conversations.append(current_conversation)
        
        return conversations

    def _process_thread_complete(self, channel_id: str, channel_name: str, msg_rows: List) -> List[Dict]:
        """Process ensuring complete thread capture regardless of size."""
        conversations = []
        thread_conversations = {}  # thread_ts -> conversation
        
        for msg_row in msg_rows:
            ts, user_id, thread_ts, subtype = msg_row
            
            if subtype in ["channel_join", "channel_leave", "bot_message"]:
                continue
                
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if not raw_data:
                continue
                
            msg_data = json.loads(raw_data)
            user_name = self.db.get_user_display_name(user_id)
            formatted_msg = self._format_message(msg_data, user_name, include_metadata=True)
            
            msg_obj = {
                "timestamp": ts,
                "user": user_name,
                "content": formatted_msg,
                "tokens": self._estimate_tokens(formatted_msg)
            }
            
            if thread_ts:
                # This is part of a thread
                if thread_ts not in thread_conversations:
                    thread_conversations[thread_ts] = {
                        "channel": channel_name,
                        "messages": [],
                        "token_count": 0,
                        "thread_ts": thread_ts
                    }
                thread_conversations[thread_ts]["messages"].append(msg_obj)
                thread_conversations[thread_ts]["token_count"] += msg_obj["tokens"]
            else:
                # Check if this starts a thread
                if msg_data.get("reply_count", 0) > 0:
                    thread_msgs = self._get_thread_messages(channel_id, ts)
                    if thread_msgs:
                        # Create a complete thread conversation
                        thread_conv = {
                            "channel": channel_name,
                            "messages": [msg_obj],
                            "token_count": msg_obj["tokens"],
                            "thread_ts": ts
                        }
                        
                        for tm in thread_msgs:
                            tm_user = self.db.get_user_display_name(tm.get("user"))
                            tm_formatted = self._format_message(tm, tm_user, include_metadata=True)
                            tm_obj = {
                                "timestamp": tm.get("ts"),
                                "user": tm_user,
                                "content": tm_formatted,
                                "tokens": self._estimate_tokens(tm_formatted)
                            }
                            thread_conv["messages"].append(tm_obj)
                            thread_conv["token_count"] += tm_obj["tokens"]
                        
                        conversations.append(thread_conv)
                else:
                    # Standalone message - group with nearby messages
                    conversations.append({
                        "channel": channel_name,
                        "messages": [msg_obj],
                        "token_count": msg_obj["tokens"]
                    })
        
        # Add any remaining thread conversations
        conversations.extend(thread_conversations.values())
        
        return conversations

    def _process_temporal_context(self, channel_id: str, channel_name: str, msg_rows: List) -> List[Dict]:
        """Process maintaining temporal context with overlapping windows."""
        conversations = []
        all_messages = []
        
        # First, collect all messages with metadata
        for msg_row in msg_rows:
            ts, user_id, thread_ts, subtype = msg_row
            
            if subtype in ["channel_join", "channel_leave", "bot_message"]:
                continue
                
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if not raw_data:
                continue
                
            msg_data = json.loads(raw_data)
            user_name = self.db.get_user_display_name(user_id)
            formatted_msg = self._format_message(msg_data, user_name, include_metadata=True)
            
            all_messages.append({
                "timestamp": float(ts),
                "user": user_name,
                "content": formatted_msg,
                "tokens": self._estimate_tokens(formatted_msg),
                "importance": self._calculate_content_importance(msg_data)
            })
        
        # Create overlapping temporal windows
        window_size = self.ENHANCED_MAX_MESSAGES
        overlap = window_size // 3  # 33% overlap
        
        for i in range(0, len(all_messages), window_size - overlap):
            window_messages = all_messages[i:i + window_size]
            if not window_messages:
                continue
                
            total_tokens = sum(msg["tokens"] for msg in window_messages)
            if total_tokens > self.KNOWLEDGE_EXTRACTION_TOKENS:
                # Split this window further
                mid_point = len(window_messages) // 2
                for sub_window in [window_messages[:mid_point], window_messages[mid_point:]]:
                    if sub_window:
                        conversations.append({
                            "channel": channel_name,
                            "messages": sub_window,
                            "token_count": sum(msg["tokens"] for msg in sub_window),
                            "temporal_window": True
                        })
            else:
                conversations.append({
                    "channel": channel_name,
                    "messages": window_messages,
                    "token_count": total_tokens,
                    "temporal_window": True
                })
        
        return conversations

    def _process_comprehensive(self, channel_id: str, channel_name: str, msg_rows: List) -> List[Dict]:
        """Comprehensive processing combining all strategies."""
        # Get results from all strategies
        knowledge_focused = self._process_knowledge_focused(channel_id, channel_name, msg_rows)
        thread_complete = self._process_thread_complete(channel_id, channel_name, msg_rows)
        temporal_context = self._process_temporal_context(channel_id, channel_name, msg_rows)
        
        # Combine and deduplicate while preserving different perspectives
        all_conversations = []
        
        # Add knowledge-focused conversations with high priority
        for conv in knowledge_focused:
            conv["strategy"] = "knowledge_focused"
            conv["priority"] = "high"
            all_conversations.append(conv)
        
        # Add complete threads
        for conv in thread_complete:
            conv["strategy"] = "thread_complete"
            conv["priority"] = "medium"
            all_conversations.append(conv)
        
        # Add temporal context
        for conv in temporal_context:
            conv["strategy"] = "temporal_context"
            conv["priority"] = "low"
            all_conversations.append(conv)
        
        return all_conversations

    def _write_enhanced_training_file(
        self,
        channel_id: str,
        channel_name: str,
        conversations: List[Dict],
        output_dir: str,
        strategy: str = "comprehensive"
    ) -> str:
        """Write enhanced training data with multiple prompt types."""
        output_path = os.path.join(output_dir, f"{channel_name}_{strategy}.jsonl")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for conv in conversations:
                # Create multiple training examples per conversation
                prompt_types = ["concept_explanation", "problem_solution", "process_documentation", "decision_rationale"]
                
                for prompt_type in prompt_types:
                    prompt_data = self._create_knowledge_extraction_prompt(conv, prompt_type)
                    
                    # Enhanced metadata
                    metadata = {
                        "channel": conv["channel"],
                        "message_count": len(conv["messages"]),
                        "token_count": conv.get("token_count", 0),
                        "strategy": conv.get("strategy", strategy),
                        "priority": conv.get("priority", "medium"),
                        "prompt_type": prompt_type,
                        "importance_score": conv.get("importance_score", 0.0),
                        "timestamp_range": [
                            conv["messages"][0]["timestamp"] if conv["messages"] else None,
                            conv["messages"][-1]["timestamp"] if conv["messages"] else None
                        ]
                    }
                    
                    training_example = {
                        "conversations": [
                            {
                                "role": "system",
                                "content": prompt_data["system"]
                            },
                            {
                                "role": "user", 
                                "content": prompt_data["user"]
                            },
                            {
                                "role": "assistant",
                                "content": prompt_data["assistant"]
                            }
                        ],
                        "metadata": metadata
                    }
                    
                    f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
        
        return output_path

    def export_for_comprehensive_training(self, output_dir: str, strategies: List[str] = None) -> None:
        """Export with comprehensive knowledge capture strategies."""
        if strategies is None:
            strategies = ["knowledge_focused", "thread_complete", "temporal_context", "comprehensive"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        channels = self._get_all_channels()
        all_files = []
        total_conversations = 0
        
        print(f"\nExporting {len(channels)} channels with enhanced strategies: {', '.join(strategies)}")
        
        for channel_id, channel_name in channels:
            try:
                print(f"Processing channel #{channel_name}...")
                
                for strategy in strategies:
                    conversations = self._process_channel_conversations_enhanced(
                        channel_id, channel_name, strategy
                    )
                    
                    if not conversations:
                        continue
                    
                    output_path = self._write_enhanced_training_file(
                        channel_id, channel_name, conversations, output_dir, strategy
                    )
                    
                    # Calculate total training examples (4 prompt types per conversation)
                    training_examples = len(conversations) * 4
                    
                    all_files.append({
                        "path": output_path,
                        "strategy": strategy,
                        "conversation_count": len(conversations),
                        "training_examples": training_examples,
                        "channel": channel_name
                    })
                    
                    total_conversations += len(conversations)
                    print(f"  {strategy}: {len(conversations)} conversations, {training_examples} training examples")
                    
            except Exception as e:
                logging.error(f"Error processing channel {channel_name}: {str(e)}")
                continue
        
        # Write comprehensive metadata
        self._write_enhanced_metadata(output_dir, total_conversations, channels, all_files, strategies)
        
        total_examples = sum(f["training_examples"] for f in all_files)
        print(f"\nEnhanced export complete:")
        print(f"  Total conversations: {total_conversations:,}")
        print(f"  Total training examples: {total_examples:,}")
        print(f"  Strategies used: {', '.join(strategies)}")

    def _write_enhanced_metadata(
        self,
        output_dir: str,
        total_conversations: int,
        channels: List[Tuple[str, str]],
        all_files: List[Dict],
        strategies: List[str]
    ) -> None:
        """Write enhanced metadata for comprehensive training."""
        summary = {
            "total_conversations": total_conversations,
            "total_training_examples": sum(f["training_examples"] for f in all_files),
            "total_channels": len(channels),
            "strategies_used": strategies,
            "channels": [{"id": ch_id, "name": ch_name} for ch_id, ch_name in channels],
            "files": all_files,
            "exported_at": datetime.now().isoformat(),
            "enhanced_format_info": {
                "knowledge_extraction_tokens": self.KNOWLEDGE_EXTRACTION_TOKENS,
                "enhanced_max_messages": self.ENHANCED_MAX_MESSAGES,
                "enhanced_max_tokens": self.ENHANCED_MAX_TOKENS,
                "prompt_types": ["concept_explanation", "problem_solution", "process_documentation", "decision_rationale"],
                "strategies": {
                    "knowledge_focused": "Prioritizes high-value technical and organizational knowledge",
                    "thread_complete": "Ensures complete thread capture for context continuity",
                    "temporal_context": "Maintains temporal relationships with overlapping windows",
                    "comprehensive": "Combines all strategies for maximum coverage"
                },
                "importance_factors": [
                    "Message length and technical keyword density",
                    "Thread engagement and reply count",
                    "File attachments and code blocks",
                    "Community reactions and validation",
                    "URLs and external references"
                ]
            }
        }
        
        with open(os.path.join(output_dir, "enhanced_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Enhanced README
        readme_content = f"""# Enhanced Slack Knowledge Training Data

This directory contains Slack conversations optimized for comprehensive organizational knowledge capture.

## Enhanced Strategies

### Knowledge-Focused Export
- Prioritizes messages with high technical/organizational value
- Uses importance scoring based on length, keywords, engagement
- Context window: {self.KNOWLEDGE_EXTRACTION_TOKENS:,} tokens
- Filters out low-value content to focus on knowledge-dense discussions

### Thread-Complete Export  
- Ensures complete thread capture regardless of size
- Maintains full context for complex discussions
- Preserves problem-solution continuity
- Ideal for troubleshooting and decision-making knowledge

### Temporal-Context Export
- Maintains temporal relationships with overlapping windows
- Captures evolving discussions and context development
- 33% overlap between windows for continuity
- Preserves chronological knowledge development

### Comprehensive Export
- Combines all strategies for maximum coverage
- Multiple perspectives on the same content
- Prioritized training examples (high/medium/low priority)
- 4x training examples per conversation (different prompt types)

## Training Data Format

Each conversation generates 4 training examples with different prompt types:

1. **Concept Explanation**: Extract and explain technical concepts
2. **Problem-Solution**: Document troubleshooting and solutions  
3. **Process Documentation**: Capture workflows and procedures
4. **Decision Rationale**: Preserve decision-making reasoning

## Optimization for GPU Cycles

This format is designed for scenarios where you want to spend extra GPU cycles for better knowledge capture:

- Larger context windows ({self.KNOWLEDGE_EXTRACTION_TOKENS:,} tokens vs 2,048)
- More training examples per conversation (4x multiplier)
- Multiple processing strategies for comprehensive coverage
- Importance-based filtering to focus on valuable content
- Complete thread preservation for complex discussions

## Statistics

- Total conversations: {total_conversations:,}
- Total training examples: {sum(f['training_examples'] for f in all_files):,}
- Total channels: {len(channels):,}
- Strategies: {', '.join(strategies)}
- Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage Recommendations

1. **For comprehensive knowledge**: Use all strategies
2. **For focused training**: Use knowledge_focused only
3. **For problem-solving**: Emphasize thread_complete
4. **For process capture**: Use temporal_context + knowledge_focused

The enhanced format trades training time for knowledge comprehensiveness, making your fine-tuned model much more capable of explaining organizational concepts that were discussed informally in Slack.
"""
        
        with open(os.path.join(output_dir, "ENHANCED_README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)

    def export_for_rag(self, output_dir: str, include_thread_summaries: bool = True, batch_size: int = 1000) -> None:
        """Export database contents optimized for RAG (Retrieval Augmented Generation).
        
        Streams the export process to handle very large databases with minimal memory overhead.
        Bot messages are automatically filtered out as they contain no useful information.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        channels = self._get_all_channels()
        output_path = os.path.join(output_dir, "slack_rag_documents.jsonl")
        
        # Statistics tracking (incremental)
        stats = {
            "total_documents": 0,
            "document_types": defaultdict(int),
            "channels": len(channels),
            "users": set(),
            "date_range": {"earliest": None, "latest": None},
            "content_types": defaultdict(int),
            "total_length": 0,
            "has_code": 0,
            "has_files": 0,
            "threads": 0,
            "bot_messages_filtered": 0,
        }
        
        print(f"\nStreaming export of {len(channels)} channels for RAG to {output_dir}...")
        print("Bot messages are automatically filtered out")
        
        # Stream documents to file, processing one channel at a time
        with open(output_path, "w", encoding="utf-8") as f:
            for channel_id, channel_name in channels:
                try:
                    print(f"Processing channel #{channel_name}...")
                    channel_doc_count = 0
                    
                    # Process channel in streaming fashion
                    for doc in self._stream_channel_for_rag(channel_id, channel_name, include_thread_summaries, batch_size, stats):
                        # Write document immediately
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        
                        # Update statistics incrementally
                        self._update_stats_incremental(stats, doc)
                        channel_doc_count += 1
                        
                        # Periodic progress update for large channels
                        if channel_doc_count % 10000 == 0:
                            print(f"  ...processed {channel_doc_count:,} documents")
                    
                    print(f"  Generated {channel_doc_count:,} documents")
                    
                except Exception as e:
                    logging.error(f"Error processing channel {channel_name}: {str(e)}")
                    continue
        
        # Finalize statistics
        self._finalize_stats(stats)
        
        # Create metadata and statistics
        self._write_rag_metadata_streaming(output_dir, stats, channels)
        
        print(f"\nRAG export complete: {stats['total_documents']:,} documents")
        print(f"Bot messages filtered: {stats['bot_messages_filtered']:,}")
        print(f"Output: {output_path}")

    def _stream_channel_for_rag(self, channel_id: str, channel_name: str, include_thread_summaries: bool, batch_size: int, stats: Dict):
        """Stream RAG-optimized documents for a channel with minimal memory overhead."""
        # Use a separate cursor for this operation to avoid conflicts
        cur = self.db.conn.cursor()
        processed_threads = set()
        
        # For very large channels, we need to be extra careful about memory
        # Use server-side cursor with LIMIT/OFFSET to handle massive channels
        offset = 0
        
        # Get total count for progress tracking (optional for very large DBs)
        try:
            cur.execute("SELECT COUNT(*) FROM messages WHERE channel_id = ? AND subtype NOT IN ('channel_join', 'channel_leave', 'bot_message')", (channel_id,))
            total_count = cur.fetchone()[0]
        except:
            total_count = None  # Skip progress tracking if count is too expensive
        
        while True:
            # Fetch messages in batches to avoid loading entire channel into memory
            cur.execute(
                """
                SELECT m.ts, m.user_id, m.thread_ts, m.subtype, m.reply_count
                FROM messages m
                WHERE m.channel_id = ?
                ORDER BY m.ts
                LIMIT ? OFFSET ?
            """,
                (channel_id, batch_size, offset),
            )
            
            batch = cur.fetchall()
            if not batch:
                break
            
            for msg_row in batch:
                ts, user_id, thread_ts, subtype, reply_count = msg_row
                
                # Skip system messages and bot messages (nobody cares about them)
                if subtype in ["channel_join", "channel_leave", "bot_message"]:
                    if subtype == "bot_message":
                        stats["bot_messages_filtered"] += 1
                    continue
                    
                # Get the full message data
                raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
                if not raw_data:
                    continue
                    
                try:
                    msg_data = json.loads(raw_data)
                    
                    # Additional bot message filtering based on message content
                    if msg_data.get("bot_id") or msg_data.get("username"):
                        stats["bot_messages_filtered"] += 1
                        continue
                    
                    user_name = self.db.get_user_display_name(user_id)
                    
                    # Yield individual message document
                    msg_doc = self._create_message_document(
                        msg_data, user_name, channel_name, channel_id, ts
                    )
                    yield msg_doc
                    
                    # Yield thread summary document if this is a thread starter
                    if (include_thread_summaries and 
                        not thread_ts and 
                        reply_count and reply_count > 2 and 
                        ts not in processed_threads):
                        
                        thread_doc = self._create_thread_document(
                            channel_id, channel_name, ts, msg_data, user_name
                        )
                        if thread_doc:
                            yield thread_doc
                            processed_threads.add(ts)
                            
                    # Explicit cleanup for large messages
                    del msg_data, raw_data
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Error processing message {channel_id}_{ts}: {e}")
                    continue
            
            offset += batch_size
            
            # Progress update for large channels
            if total_count and offset % (batch_size * 10) == 0:
                progress = min(100, (offset / total_count) * 100)
                print(f"    Progress: {progress:.1f}% ({offset:,}/{total_count:,})")
            
            # Clear batch from memory explicitly
            del batch
            
        # Clean up processed threads set
        processed_threads.clear()
        cur.close()

    def _create_message_document(self, msg_data: Dict, user_name: str, channel_name: str, channel_id: str, ts: str) -> Dict:
        """Create a RAG document for an individual message."""
        text = msg_data.get("text", "").strip()
        
        # Clean up Slack formatting for better semantic search
        text = self._clean_slack_formatting(text)
        
        # Extract mentioned users, channels, and links
        mentions = self._extract_mentions(msg_data.get("text", ""))
        
        # Determine content type and importance
        content_type, importance_score = self._analyze_content_type(msg_data, text)
        
        # Create timestamp
        dt = datetime.fromtimestamp(float(ts))
        
        # Build the document
        document = {
            "id": f"{channel_id}_{ts}",
            "content": text,
            "metadata": {
                "type": "message",
                "channel": channel_name,
                "channel_id": channel_id,
                "user": user_name,
                "user_id": msg_data.get("user"),
                "timestamp": ts,
                "datetime": dt.isoformat(),
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "day_of_week": dt.strftime("%A"),
                "content_type": content_type,
                "importance_score": importance_score,
                "thread_ts": msg_data.get("thread_ts"),
                "has_thread": bool(msg_data.get("reply_count", 0) > 0),
                "reply_count": msg_data.get("reply_count", 0),
                "mentions": mentions,
                "has_files": bool(msg_data.get("files")),
                "has_code": "```" in text or text.count("`") > 2,
                "has_links": "http" in text.lower(),
                "reactions": [r.get("name") for r in msg_data.get("reactions", [])],
                "reaction_count": sum(r.get("count", 0) for r in msg_data.get("reactions", [])),
                "edited": bool(msg_data.get("edited")),
                "char_length": len(text),
                "word_count": len(text.split()) if text else 0,
            }
        }
        
        # Add file information if present
        if msg_data.get("files"):
            file_info = []
            for f in msg_data.get("files", []):
                file_info.append({
                    "name": f.get("name", ""),
                    "type": f.get("filetype", ""),
                    "size": f.get("size", 0),
                    "title": f.get("title", "")
                })
            document["metadata"]["files"] = file_info
            
            # Add file content to main content for search
            file_descriptions = [f"[File: {f['name']} ({f.get('type', 'unknown')})]" for f in file_info]
            if file_descriptions:
                document["content"] += "\n" + "\n".join(file_descriptions)
        
        return document

    def _create_thread_document(self, channel_id: str, channel_name: str, thread_ts: str, original_msg: Dict, original_user: str) -> Optional[Dict]:
        """Create a RAG document for an entire thread discussion."""
        thread_messages = self._get_thread_messages(channel_id, thread_ts)
        if not thread_messages:
            return None
            
        # Combine all thread messages
        all_messages = [original_msg] + thread_messages
        participants = set()
        full_conversation = []
        
        for msg in all_messages:
            user_id = msg.get("user")
            if user_id:
                user_name = self.db.get_user_display_name(user_id)
                participants.add(user_name)
                text = self._clean_slack_formatting(msg.get("text", ""))
                if text:
                    full_conversation.append(f"{user_name}: {text}")
        
        if not full_conversation:
            return None
            
        # Create thread summary
        conversation_text = "\n".join(full_conversation)
        original_text = self._clean_slack_formatting(original_msg.get("text", ""))
        
        # Extract key topics/keywords from the thread
        thread_keywords = self._extract_thread_keywords(conversation_text)
        
        dt = datetime.fromtimestamp(float(thread_ts))
        
        return {
            "id": f"{channel_id}_thread_{thread_ts}",
            "content": f"Thread Discussion:\n\nOriginal: {original_text}\n\nConversation:\n{conversation_text}",
            "metadata": {
                "type": "thread",
                "channel": channel_name,
                "channel_id": channel_id,
                "thread_starter": original_user,
                "participants": sorted(list(participants)),
                "participant_count": len(participants),
                "message_count": len(all_messages),
                "timestamp": thread_ts,
                "datetime": dt.isoformat(),
                "date": dt.strftime("%Y-%m-%d"),
                "keywords": thread_keywords,
                "importance_score": min(len(participants) * 0.5 + len(all_messages) * 0.2, 5.0),
                "char_length": len(conversation_text),
                "word_count": len(conversation_text.split()),
            }
        }

    def _clean_slack_formatting(self, text: str) -> str:
        """Clean Slack-specific formatting for better semantic search."""
        if not text:
            return ""
            
        # Convert user mentions to readable format
        text = re.sub(r'<@([A-Z0-9]+)>', lambda m: f"@{self.db.get_user_display_name(m.group(1))}", text)
        
        # Convert channel mentions
        text = re.sub(r'<#([A-Z0-9]+)\|([^>]+)>', r'#\2', text)
        
        # Convert links
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _extract_mentions(self, text: str) -> Dict[str, List[str]]:
        """Extract different types of mentions from message text."""
        mentions = {
            "users": [],
            "channels": [],
            "links": []
        }
        
        if not text:
            return mentions
            
        # Extract user mentions
        user_matches = re.findall(r'<@([A-Z0-9]+)>', text)
        mentions["users"] = [self.db.get_user_display_name(uid) for uid in user_matches]
        
        # Extract channel mentions
        channel_matches = re.findall(r'<#[A-Z0-9]+\|([^>]+)>', text)
        mentions["channels"] = channel_matches
        
        # Extract links
        link_matches = re.findall(r'<(https?://[^|>]+)', text)
        mentions["links"] = link_matches
        
        return mentions

    def _analyze_content_type(self, msg_data: Dict, text: str) -> Tuple[str, float]:
        """Analyze message content to determine type and importance."""
        content_types = []
        importance = 1.0
        
        # Check for different content indicators
        if "```" in text or text.count("`") > 2:
            content_types.append("code")
            importance += 1.5
            
        if msg_data.get("files"):
            content_types.append("file_share")
            importance += 1.0
            
        if msg_data.get("reply_count", 0) > 0:
            content_types.append("discussion_starter")
            importance += msg_data["reply_count"] * 0.3
            
        if any(keyword in text.lower() for keyword in ["decision", "concluded", "agreed", "resolved"]):
            content_types.append("decision")
            importance += 2.0
            
        if any(keyword in text.lower() for keyword in ["problem", "issue", "bug", "error", "broken"]):
            content_types.append("problem_report")
            importance += 1.5
            
        if any(keyword in text.lower() for keyword in ["solution", "fix", "resolved", "working"]):
            content_types.append("solution")
            importance += 1.5
            
        if any(keyword in text.lower() for keyword in ["architecture", "design", "process", "workflow"]):
            content_types.append("technical_discussion")
            importance += 1.0
            
        if len(text) > 500:
            content_types.append("detailed_explanation")
            importance += 0.5
            
        if msg_data.get("reactions"):
            content_types.append("community_validated")
            importance += sum(r.get("count", 0) for r in msg_data["reactions"]) * 0.1
        
        return "|".join(content_types) if content_types else "general", min(importance, 5.0)

    def _extract_thread_keywords(self, conversation_text: str) -> List[str]:
        """Extract key topics and keywords from thread conversation."""
        # Simple keyword extraction - could be enhanced with NLP
        text = conversation_text.lower()
        
        # Technical keywords
        tech_keywords = []
        for keyword in self.TECHNICAL_KEYWORDS:
            if keyword in text:
                tech_keywords.append(keyword)
        
        # System/product names (capitalize words that appear frequently)
        words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b', conversation_text)
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] += 1
        
        # Get frequently mentioned proper nouns
        frequent_terms = [word for word, count in word_freq.items() if count > 1]
        
        return sorted(list(set(tech_keywords + frequent_terms)))

    def _update_stats_incremental(self, stats: Dict, doc: Dict) -> None:
        """Update statistics incrementally for streaming export."""
        meta = doc["metadata"]
        
        stats["total_documents"] += 1
        stats["document_types"][meta["type"]] += 1
        stats["users"].add(meta.get("user", ""))
        stats["content_types"][meta.get("content_type", "")] += 1
        
        if meta.get("has_code"):
            stats["has_code"] += 1
        if meta.get("has_files"):
            stats["has_files"] += 1
        if meta["type"] == "thread":
            stats["threads"] += 1
            
        doc_len = meta.get("char_length", 0)
        stats["total_length"] += doc_len
        
        # Track date range
        doc_date = meta.get("datetime")
        if doc_date:
            if not stats["date_range"]["earliest"] or doc_date < stats["date_range"]["earliest"]:
                stats["date_range"]["earliest"] = doc_date
            if not stats["date_range"]["latest"] or doc_date > stats["date_range"]["latest"]:
                stats["date_range"]["latest"] = doc_date

    def _finalize_stats(self, stats: Dict) -> None:
        """Finalize statistics after streaming export."""
        stats["avg_length"] = stats["total_length"] / stats["total_documents"] if stats["total_documents"] else 0
        stats["users"] = sorted(list(stats["users"]))
        stats["user_count"] = len(stats["users"])
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["document_types"] = dict(stats["document_types"])
        stats["content_types"] = dict(stats["content_types"])

    def _write_rag_metadata_streaming(self, output_dir: str, stats: Dict, channels: List[Tuple[str, str]]) -> None:
        """Write metadata files for RAG export using precomputed streaming stats."""
        
        metadata = {
            "export_type": "rag_streaming",
            "exported_at": datetime.now().isoformat(),
            "statistics": stats,
            "channels": [{"id": ch_id, "name": ch_name} for ch_id, ch_name in channels],
            "schema": {
                "document_structure": {
                    "id": "Unique document identifier",
                    "content": "Main searchable content",
                    "metadata": "Rich metadata for filtering and attribution"
                },
                "metadata_fields": {
                    "type": "message or thread",
                    "channel": "Channel name",
                    "user": "User display name",
                    "timestamp": "Unix timestamp",
                    "datetime": "ISO datetime",
                    "content_type": "Classified content type",
                    "importance_score": "Computed relevance score (1-5)",
                    "mentions": "Extracted user/channel/link mentions",
                    "has_code": "Contains code blocks",
                    "has_files": "Contains file attachments",
                    "reactions": "Reaction emojis received",
                    "participants": "Thread participants (threads only)",
                    "keywords": "Extracted keywords (threads only)"
                }
            },
            "usage_recommendations": {
                "semantic_search": "Use 'content' field for main semantic search",
                "attribution_search": "Filter by 'user' in metadata",
                "temporal_search": "Filter by 'date' or 'datetime' in metadata",
                "content_type_search": "Filter by 'content_type' for specific discussion types",
                "importance_filtering": "Use 'importance_score' to prioritize high-value content",
                "thread_reconstruction": "Use 'thread_ts' to group related messages"
            }
        }
        
        # Write metadata
        with open(os.path.join(output_dir, "rag_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Write README for RAG usage
        readme_content = f"""# Slack RAG Export (Memory-Efficient Streaming)

This directory contains Slack conversations optimized for Retrieval Augmented Generation (RAG).

**OPTIMIZED FOR TERABYTE-SCALE DATABASES**: This export uses streaming processing with configurable batch sizes to handle massive datasets with minimal memory overhead.

**BOT MESSAGES FILTERED**: Bot messages are automatically filtered out as they contain no useful information for RAG.

## Memory-Efficient Features

- **Streaming Processing**: Documents are processed and written one-by-one, never loading entire datasets into memory
- **Configurable Batch Size**: Use `--batch-size` to control memory usage (default: 1000 messages per batch)
- **Progress Tracking**: Real-time progress updates for large channels
- **Explicit Memory Management**: Automatic cleanup of processed data to prevent memory leaks
- **Channel-by-Channel Processing**: Each channel is processed independently to maintain low memory footprint

## Usage for Large Databases

```bash
# For very large databases (terabytes), use smaller batch sizes:
python app.py --rag ./rag_export --batch-size 100

# For systems with more RAM, you can use larger batches for faster processing:
python app.py --rag ./rag_export --batch-size 5000

# Default (balanced for most systems):
python app.py --rag ./rag_export --batch-size 1000
```

## Document Structure

Each document in `slack_rag_documents.jsonl` has this structure:

```json
{{
    "id": "unique_document_id",
    "content": "searchable text content",
    "metadata": {{
        "type": "message|thread",
        "channel": "channel-name",
        "user": "User Name",
        "timestamp": "1234567890.123456",
        "datetime": "2024-01-01T12:00:00",
        "content_type": "code|discussion_starter|decision|etc",
        "importance_score": 2.5,
        "mentions": {{"users": [...], "channels": [...], "links": [...]}},
        "has_code": true,
        "has_files": false,
        "reactions": ["thumbsup", "eyes"],
        // ... additional metadata
    }}
}}
```

## Statistics

- **Total Documents**: {stats['total_documents']:,}
- **Messages**: {stats['document_types'].get('message', 0):,}
- **Thread Summaries**: {stats['document_types'].get('thread', 0):,}
- **Unique Users**: {stats['user_count']:,}
- **Channels**: {stats['channels']:,}
- **Bot Messages Filtered**: {stats['bot_messages_filtered']:,}
- **Date Range**: {stats['date_range']['earliest'][:10] if stats['date_range']['earliest'] else 'N/A'} to {stats['date_range']['latest'][:10] if stats['date_range']['latest'] else 'N/A'}
- **Avg Document Length**: {stats['avg_length']:.0f} characters
- **Documents with Code**: {stats['has_code']:,}
- **Documents with Files**: {stats['has_files']:,}

## RAG Usage Patterns

### Attribution Queries
Find what specific people said about topics:
```
Query: "What does Andrea think about product strategy?"
Filter: metadata.user = "Andrea"
```

### System/Technical Queries  
Find discussions about specific systems:
```
Query: "Back Office ChartFinder interaction"
Filter: metadata.content_type contains "technical_discussion"
```

### Expert Knowledge Queries
Find expertise from specific people:
```
Query: "SQL recommendations for lost records"  
Filter: metadata.user = "Josh" AND metadata.has_code = true
```

### Temporal Queries
Find discussions from specific time periods:
```
Filter: metadata.date >= "2024-01-01"
```

### High-Value Content
Focus on important discussions:
```
Filter: metadata.importance_score >= 3.0
```

## Framework Integration

This format works with most RAG frameworks:

- **LangChain**: Use `JSONLinesLoader` to load documents
- **LlamaIndex**: Use `JSONReader` with metadata support  
- **Haystack**: Use `JsonlDocumentStore`
- **Chroma**: Load as documents with metadata
- **Pinecone/Weaviate**: Use metadata for filtering

## Content Types

The export identifies these content types:
{chr(10).join(f"- **{k}**: {v} documents" for k, v in sorted(stats['content_types'].items()) if k)}

See `rag_metadata.json` for complete statistics and schema information.
"""
        
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Archive Slack channels via in-browser endpoints"
    )

    # Create a mutually exclusive group for the different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "subdomain",
        nargs="?",
        help="Slack workspace subdomain (e.g. datavant.enterprise)",
    )

    mode_group.add_argument(
        "--enhanced-export",
        help="Export with enhanced knowledge capture strategies to the specified directory",
    )
    mode_group.add_argument(
        "--rag",
        help="Export database contents optimized for RAG (Retrieval Augmented Generation) to the specified directory",
    )

    # Enhanced export options
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["knowledge_focused", "thread_complete", "temporal_context", "comprehensive"],
        default=["comprehensive"],
        help="Export strategies to use (default: comprehensive). Can specify multiple.",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing messages during RAG export (default: 1000, lower values use less memory)",
    )

    args = parser.parse_args()

    # For archive mode, validate required arguments
    if not args.enhanced_export and not args.rag and not all(
        [args.subdomain, args.org, args.x_version_timestamp, args.team, args.cookie]
    ):
        parser.error(
            "When archiving (not using --enhanced-export or --rag), the following arguments are required: subdomain, --org, --x-version-timestamp, --team, --cookie"
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
        if args.enhanced_export:
            exporter = Exporter(db)
            exporter.export_for_comprehensive_training(args.enhanced_export, args.strategies)
            return
        
        if args.rag:
            exporter = Exporter(db)
            exporter.export_for_rag(args.rag, include_thread_summaries=True, batch_size=args.batch_size)
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
        channels_processed, messages_processed = message_manager.sync_all_channels()

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
