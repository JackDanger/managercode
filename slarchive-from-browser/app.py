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
from exporter import Exporter

DB_PATH = "./db.sqlite3"


class DatabaseManager:
    """Manages database operations and compression."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.usernames = {}

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

        if user_id in self.usernames:
            return self.usernames[user_id]

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
        value = display_name or real_name or name or f"<@{user_id}>"
        self.usernames[user_id] = value
        return value

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

        # Create indexes separately to handle multiple statements
        index_statements = self._get_messages_index_ddl().strip().split(';')
        for statement in index_statements:
            statement = statement.strip()
            if statement:  # Skip empty statements
                cur.execute(statement)

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
    def _get_messages_index_ddl(self) -> str:
        return """
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(channel_id, thread_ts, ts);
        CREATE INDEX IF NOT EXISTS idx_messages_subtype_filter ON messages(channel_id, ts) WHERE subtype NOT IN ('channel_join', 'channel_leave', 'bot_message');
        CREATE INDEX IF NOT EXISTS idx_messages_reply_count ON messages(channel_id, reply_count, ts) WHERE reply_count > 0;
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

    def migrate_indexes(self) -> None:
        """
        Migrate existing database to use optimized indexes.
        This method can be called to update databases created with the old index structure.
        """
        cur = self.conn.cursor()

        # Drop the old inefficient index if it exists
        try:
            cur.execute("DROP INDEX IF EXISTS channel_subtype_ts_idx")
            print("Dropped old inefficient index: channel_subtype_ts_idx")
        except Exception as e:
            logging.warning(f"Could not drop old index: {e}")

        # Create new optimized indexes
        index_statements = self._get_messages_index_ddl().strip().split(';')
        for statement in index_statements:
            statement = statement.strip()
            if statement:
                try:
                    cur.execute(statement)
                    # Extract index name for logging
                    import re
                    match = re.search(r'CREATE INDEX IF NOT EXISTS (\w+)', statement)
                    if match:
                        print(f"Created optimized index: {match.group(1)}")
                except Exception as e:
                    logging.error(f"Failed to create index: {statement}\nError: {e}")

        self.conn.commit()
        print("Index migration completed!")


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        "--fine-tune",
        help="Export database contents for fine-tuning to the specified directory",
    )
    mode_group.add_argument(
        "--rag",
        help="Export database contents optimized for RAG (Retrieval Augmented Generation) to the specified directory",
    )
    mode_group.add_argument(
        "--migrate-indexes",
        action="store_true",
        help="Migrate existing database to use optimized indexes (recommended for existing installations)",
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
    if not args.fine_tune and not args.rag and not args.migrate_indexes and not all(
        [args.subdomain, args.org, args.x_version_timestamp, args.team, args.cookie]
    ):
        parser.error(
            "When archiving (not using --fine-tune, --rag, or --migrate-indexes), the following arguments are required: subdomain, --org, --x-version-timestamp, --team, --cookie"
        )

    return args


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    db = DatabaseManager()
    db.connect()

    try:
        if args.migrate_indexes:
            print("\n=== Migrating Database Indexes ===\n")
            db.migrate_indexes()
            return

        if args.fine_tune:
            exporter = Exporter(db)
            exporter.export_for_fine_tuning(args.fine_tune)
            return

        if args.rag:
            exporter = Exporter(db)
            exporter.export_for_rag(args.rag, include_thread_summaries=True, batch_size=args.batch_size)
            return

        importer = Importer(db, args)
        importer.import_slack_archive()


    finally:
        db.close()


if __name__ == "__main__":
    main()
