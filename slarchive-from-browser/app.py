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
            
    def _setup_schema(self) -> None:
        """Create database tables if they don't exist."""
        cur = self.conn.cursor()
        
        # Create all tables
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
        cur.execute("""
            SELECT compressed_data 
            FROM compressed_data 
            WHERE table_name = ? AND record_id = ?
        """, (table_name, record_id))
        
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
            
            for channel in channels_to_sync:
                messages = self._fetch_channel_messages(channel)
                messages_processed += len(messages)
                channels_processed += 1
                
                self._update_sync_progress(sync_run_id, channels_processed, messages_processed)
                time.sleep(self.slack.config.rate_limit)
                
            self._complete_sync_run(sync_run_id, channels_processed, messages_processed)
            return channels_processed, messages_processed
            
        except Exception as e:
            self._fail_sync_run(sync_run_id, str(e))
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
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                    
                logging.warning(f"Error fetching users (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(self.slack.config.rate_limit * 2)
                
        return users_processed

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
