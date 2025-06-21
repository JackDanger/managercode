import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
from app import DatabaseManager


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


class Importer:
    """Handles Slack archive import operations."""

    def __init__(self, db, args):
        self.db = db
        self.args = args

    def import_slack_archive(self):
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

