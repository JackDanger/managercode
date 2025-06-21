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
        cur.execute("""
            SELECT real_name, display_name, name
            FROM users
            WHERE id = ?
        """, (user_id,))
        
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS db_version (
                version INTEGER PRIMARY KEY,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create all other tables
        tables = [
            self._get_sync_state_schema(),
            self._get_channels_schema(),
            self._get_messages_schema(),
            self._get_users_schema(),
            self._get_sync_runs_schema(),
            self._get_compressed_data_schema()
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
        cur.execute("INSERT INTO db_version (version) VALUES (?)", (self.CURRENT_VERSION,))
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
                self.store_compressed_data('channels', ch_id, raw_json)
                cur.execute("UPDATE channels SET raw_json = NULL WHERE id = ?", (ch_id,))
        
        # Migrate messages
        logging.info("Migrating messages...")
        cur.execute("SELECT channel_id, ts, raw_json FROM messages WHERE raw_json IS NOT NULL")
        for ch_id, ts, raw_json in cur.fetchall():
            if raw_json:
                self.store_compressed_data('messages', f"{ch_id}_{ts}", raw_json)
                cur.execute(
                    "UPDATE messages SET raw_json = NULL WHERE channel_id = ? AND ts = ?",
                    (ch_id, ts)
                )
        
        # Migrate users
        logging.info("Migrating users...")
        cur.execute("SELECT id, raw_json FROM users WHERE raw_json IS NOT NULL")
        for user_id, raw_json in cur.fetchall():
            if raw_json:
                self.store_compressed_data('users', user_id, raw_json)
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
            data = data.encode('utf-8')
        return gzip.compress(data)
        
    def decompress_data(self, compressed_data: bytes) -> Optional[str]:
        """Decompress gzip data."""
        if compressed_data is None:
            return None
        return gzip.decompress(compressed_data).decode('utf-8')
        
    def store_compressed_data(self, table_name: str, record_id: str, data: str) -> None:
        """Store compressed data in the compressed_data table."""
        if not data:
            return
            
        compressed = self.compress_data(data)
        original_size = len(data.encode('utf-8') if isinstance(data, str) else data)
        compressed_size = len(compressed)
        
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO compressed_data 
            (table_name, record_id, compressed_data, original_size, compressed_size, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (table_name, record_id, compressed, original_size, compressed_size))
        self.conn.commit()
        
    def get_compressed_data(self, table_name: str, record_id: str) -> Optional[str]:
        """Retrieve and decompress data from the compressed_data table."""
        cur = self.conn.cursor()
        
        # First try compressed data
        cur.execute("""
            SELECT compressed_data 
            FROM compressed_data 
            WHERE table_name = ? AND record_id = ?
        """, (table_name, record_id))
        
        row = cur.fetchone()
        if row:
            return self.decompress_data(row[0])
            
        # If not found in compressed_data, try raw_json (for backward compatibility)
        if table_name == 'channels':
            cur.execute("SELECT raw_json FROM channels WHERE id = ?", (record_id,))
        elif table_name == 'messages':
            ch_id, ts = record_id.split('_')
            cur.execute("SELECT raw_json FROM messages WHERE channel_id = ? AND ts = ?", (ch_id, ts))
        elif table_name == 'users':
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
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SlackArchiver/1.0)",
            "Cookie": self.config.cookie,
        })
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
        
    def fetch_all_channels(self) -> int:
        """Fetch all channels from Slack."""
        cur = self.db.conn.cursor()
        page = 1
        per_page = 50
        channels_processed = 0
        
        while True:
            items = self._fetch_channel_page(page, per_page)
            if not items:
                break
                
            for channel in items:
                self._process_channel(channel)
                channels_processed += 1
                
            self.db.conn.commit()
            if len(items) < per_page:
                break
                
            page += 1
            time.sleep(self.slack.config.rate_limit)
            
        return channels_processed
        
    def _fetch_channel_page(self, page: int, per_page: int) -> List[Dict]:
        """Fetch a single page of channels."""
        url = f"https://{self.slack.config.subdomain}.slack.com/api/search.modules.channels"
        data = self._build_channel_request_data(page, per_page)
        headers = self._build_channel_request_headers()
        params = self._build_channel_request_params()
        
        if self.slack.config.verbose:
            logging.debug(f"POST {url}  page={page}")
            
        r = self.slack.session.post(url, headers=headers, data=data, params=params)
        r.raise_for_status()
        return r.json().get("items", [])
        
    def _process_channel(self, channel: Dict) -> None:
        """Process and store a single channel."""
        ch_id = channel.get("id") or channel.get("channel", {}).get("id")
        ch_name = channel.get("name") or channel.get("channel", {}).get("name")
        raw = json.dumps(channel, separators=(",", ":"))
        
        cur = self.db.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO channels (id, name, raw_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (ch_id, ch_name, None))
        
        self.db.store_compressed_data('channels', ch_id, raw)
        
        cur.execute("""
            INSERT OR IGNORE INTO sync_state (channel_id, is_fully_synced)
            VALUES (?, 0)
        """, (ch_id,))

class MessageManager:
    """Manages message-related operations."""
    
    def __init__(self, db: DatabaseManager, slack: SlackClient):
        self.db = db
        self.slack = slack
        
    def fetch_channel_history(self) -> Tuple[int, int]:
        """Fetch message history for all channels."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO sync_runs (status) VALUES ('in_progress')")
        sync_run_id = cur.lastrowid
        self.db.conn.commit()
        
        try:
            channels_to_sync = self._get_channels_to_sync()
            channels_processed = 0
            messages_processed = 0
            
            for ch_id, ch_name, last_sync_ts, last_cursor in channels_to_sync:
                if self.slack.config.verbose:
                    logging.info(f"Fetching history for #{ch_name} ({ch_id})")
                    if last_sync_ts:
                        logging.info(f"  Resuming from {last_sync_ts}")
                
                cursor = last_cursor
                while True:
                    messages = self._fetch_channel_messages(ch_id, cursor)
                    if not messages:
                        break
                        
                    messages_processed += self._process_messages(ch_id, messages)
                    
                    # Update sync state
                    if messages:
                        oldest_ts = min(m.get("ts") for m in messages)
                        cur.execute("""
                            INSERT OR REPLACE INTO sync_state
                            (channel_id, last_sync_ts, last_sync_cursor, updated_at)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (ch_id, oldest_ts, cursor))
                        self.db.conn.commit()
                    
                    cursor = self._get_next_cursor()
                    if not cursor:
                        # Mark channel as fully synced if we've reached the end
                        cur.execute("""
                            UPDATE sync_state
                            SET is_fully_synced = 1, updated_at = CURRENT_TIMESTAMP
                            WHERE channel_id = ?
                        """, (ch_id,))
                        self.db.conn.commit()
                        break
                        
                    time.sleep(self.slack.config.rate_limit)
                
                channels_processed += 1
                
                # Update sync run progress
                cur.execute("""
                    UPDATE sync_runs
                    SET channels_processed = ?, messages_processed = ?
                    WHERE id = ?
                """, (channels_processed, messages_processed, sync_run_id))
                self.db.conn.commit()
                
                time.sleep(self.slack.config.rate_limit)
            
            # Mark sync run as complete
            cur.execute("""
                UPDATE sync_runs
                SET status = 'completed',
                    finished_at = CURRENT_TIMESTAMP,
                    channels_processed = ?,
                    messages_processed = ?
                WHERE id = ?
            """, (channels_processed, messages_processed, sync_run_id))
            self.db.conn.commit()
            
            return channels_processed, messages_processed
            
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
            self.db.conn.commit()
            raise
            
    def _get_channels_to_sync(self) -> List[Tuple]:
        """Get list of channels that need syncing."""
        cur = self.db.conn.cursor()
        cur.execute("""
            SELECT c.id, c.name, s.last_sync_ts, s.last_sync_cursor
            FROM channels c
            LEFT JOIN sync_state s ON c.id = s.channel_id
            WHERE s.is_fully_synced = 0 OR s.is_fully_synced IS NULL
            ORDER BY s.last_sync_ts ASC NULLS FIRST
        """)
        return cur.fetchall()
        
    def _fetch_channel_messages(self, channel_id: str, cursor: Optional[str] = None) -> List[Dict]:
        """Fetch a page of messages from a channel."""
        url = f"https://{self.slack.config.subdomain}.slack.com/api/conversations.history"
        params = {
            "token": self.slack.token,
            "channel": channel_id,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
            
        if self.slack.config.verbose:
            logging.debug(f"GET {url}  cursor={cursor}")
            
        r = self.slack.session.get(url, params=params)
        r.raise_for_status()
        self._last_response = r.json()
        return self._last_response.get("messages", [])
        
    def _process_messages(self, channel_id: str, messages: List[Dict]) -> int:
        """Process and store a list of messages."""
        messages_processed = 0
        cur = self.db.conn.cursor()
        
        # Start transaction for bulk insert
        self.db.conn.execute("BEGIN TRANSACTION")
        
        try:
            for msg in messages:
                ts = msg.get("ts")
                if not ts:
                    continue
                    
                raw = json.dumps(msg, separators=(",", ":"))
                
                # Store message data
                cur.execute("""
                    INSERT OR REPLACE INTO messages
                    (channel_id, ts, thread_ts, user_id, subtype, client_msg_id,
                     edited_ts, edited_user, reply_count, reply_users_count,
                     latest_reply, is_locked, has_files, has_blocks, raw_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    channel_id,
                    ts,
                    msg.get("thread_ts"),
                    msg.get("user"),
                    msg.get("subtype"),
                    msg.get("client_msg_id"),
                    msg.get("edited", {}).get("ts"),
                    msg.get("edited", {}).get("user"),
                    msg.get("reply_count"),
                    msg.get("reply_users_count"),
                    msg.get("latest_reply"),
                    bool(msg.get("is_locked")),
                    bool(msg.get("files")),
                    bool(msg.get("blocks")),
                    None  # raw_json will be stored in compressed_data
                ))
                
                self.db.store_compressed_data('messages', f"{channel_id}_{ts}", raw)
                messages_processed += 1
                
            self.db.conn.commit()
            
        except Exception as e:
            self.db.conn.rollback()
            logging.error(f"Error processing messages: {str(e)}")
            raise
            
        return messages_processed
        
    def _get_next_cursor(self) -> Optional[str]:
        """Get the next cursor from the response."""
        if not hasattr(self, '_last_response'):
            return None
        return self._last_response.get("response_metadata", {}).get("next_cursor")

class UserManager:
    """Manages user-related operations."""
    
    def __init__(self, db: DatabaseManager, slack: SlackClient):
        self.db = db
        self.slack = slack
        
    def fetch_all_users(self) -> int:
        """Fetch all users from Slack."""
        marker = None
        users_processed = 0
        retry_count = 0
        max_retries = 3
        
        while True:
            try:
                users = self._fetch_user_page(marker)
                if not users:
                    break
                    
                users_processed += self._process_users(users)
                marker = self._get_next_marker()
                
                if not marker:
                    break
                    
                time.sleep(self.slack.config.rate_limit)
                retry_count = 0  # Reset retry count on success
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Failed to fetch users after {max_retries} retries: {str(e)}")
                    raise
                    
                logging.warning(f"Error fetching users (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(self.slack.config.rate_limit * 2)  # Exponential backoff
                
        if self.slack.config.verbose:
            logging.info(f"Processed {users_processed} users")
            
        return users_processed
        
    def _fetch_user_page(self, marker: Optional[str] = None) -> List[Dict]:
        """Fetch a single page of users from Slack."""
        url = f"https://edgeapi.slack.com/cache/{self.slack.config.org_id}/users/list"
        params = {
            "_x_app_name": "client",
            "fp": "c7",
            "_x_num_retries": "0"
        }
        
        data = {
            "token": self.slack.token,
            "count": 1000,
            "present_first": True,
            "enterprise_token": self.slack.token
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
            "sec-fetch-site": "same-site"
        }
        
        if self.slack.config.verbose:
            logging.debug(f"POST {url}  marker={marker}")
            
        r = self.slack.session.post(url, params=params, headers=headers, json=data)
        r.raise_for_status()
        self._last_response = r.json()  # Store the response for pagination
        return self._last_response.get("results", [])
        
    def _process_users(self, users: List[Dict]) -> int:
        """Process and store a list of users."""
        users_processed = 0
        cur = self.db.conn.cursor()
        
        # Start transaction for bulk insert
        self.db.conn.execute("BEGIN TRANSACTION")
        
        try:
            for user in users:
                user_id = user.get("id")
                if not user_id:
                    continue
                    
                # Validate required fields
                name = user.get("name", "").strip()
                real_name = user.get("real_name", "").strip()
                display_name = user.get("profile", {}).get("display_name", "").strip()
                
                if not any([name, real_name, display_name]):
                    logging.warning(f"Skipping user {user_id} - no valid name found")
                    continue
                    
                # Store user data
                raw = json.dumps(user, separators=(",", ":"))
                cur.execute("""
                    INSERT OR REPLACE INTO users 
                    (id, name, real_name, display_name, raw_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    user_id,
                    name,
                    real_name,
                    display_name,
                    None  # raw_json will be stored in compressed_data
                ))
                
                self.db.store_compressed_data('users', user_id, raw)
                users_processed += 1
                
            self.db.conn.commit()
            
        except Exception as e:
            self.db.conn.rollback()
            logging.error(f"Error processing users: {str(e)}")
            raise
            
        return users_processed
        
    def _get_next_marker(self) -> Optional[str]:
        """Get the next marker from the response."""
        if not hasattr(self, '_last_response'):
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
                conversations = self._process_channel_conversations(channel_id, channel_name)
                if not conversations:
                    continue
                    
                output_path = self._write_channel_file(channel_id, channel_name, conversations, output_dir)
                channel_files.append({"path": output_path, "type": "conversation"})
                total_conversations += len(conversations)
                
            except Exception as e:
                logging.error(f"Error processing channel {channel_name}: {str(e)}")
                continue
                
        self._write_metadata_files(output_dir, total_conversations, channels, channel_files)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Archive Slack channels via in-browser endpoints")
    
    # Create a mutually exclusive group for the two modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "subdomain", nargs="?", help="Slack workspace subdomain (e.g. datavant.enterprise)"
    )
    mode_group.add_argument(
        "--dump-axolotl",
        help="Export database contents for Axolotl fine-tuning to the specified directory"
    )
    
    # Add all other arguments as optional
    parser.add_argument(
        "--org", help="Which specific Slack org the cookie is signed into"
    )
    parser.add_argument(
        "--x-version-timestamp",
        help="X-Version-Timestamp from the Slack homepage"
    )
    parser.add_argument(
        "--team", help="Which specific Slack team the cookie is signed into"
    )
    parser.add_argument(
        "--cookie", help="Your full Slack session cookie string"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Seconds to wait between HTTP requests",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--client-req-id",
        help="Client request ID for API calls"
    )
    parser.add_argument(
        "--browse-session-id",
        help="Browse session ID for API calls"
    )
    
    args = parser.parse_args()
    
    # For archive mode, validate required arguments
    if not args.dump_axolotl and not all([args.subdomain, args.org, args.x_version_timestamp, args.team, args.cookie]):
        parser.error("When archiving (not using --dump-axolotl), the following arguments are required: subdomain, --org, --x-version-timestamp, --team, --cookie")
    
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
        browse_session_id=args.browse_session_id
    )

def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
        
        user_manager.fetch_all_users()
        channel_manager.fetch_all_channels()
        message_manager.fetch_channel_history()
        
        logging.info("Done! Archive stored in %s", DB_PATH)
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
