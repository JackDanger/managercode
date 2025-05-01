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
from datetime import datetime
from collections import defaultdict

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

    # Track users
    cur.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id         TEXT PRIMARY KEY,
        name       TEXT,
        real_name  TEXT,
        display_name TEXT,
        raw_json   TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def fetch_all_users(session, org_id, token, rate_limit, verbose, db_conn):
    """Fetch all users from Slack, handling pagination."""
    cur = db_conn.cursor()
    marker = None
    users_processed = 0
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            # Prepare the request
            url = f"https://edgeapi.slack.com/cache/{org_id}/users/list"
            params = {
                "_x_app_name": "client",
                "fp": "c7",
                "_x_num_retries": "0"
            }
            
            data = {
                "token": token,
                "count": 1000,
                "present_first": True,
                "enterprise_token": token
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
            
            if verbose:
                logging.debug(f"POST {url}  marker={marker}")
            
            # print_curl_command("POST", url, headers, params=params, data=data, session_headers=session.headers)
            r = session.post(url, params=params, headers=headers, json=data)
            r.raise_for_status()
            j = r.json()
            
            # Process users
            users = j.get("results", [])
            if verbose:
                logging.debug(f"Got {len(users)} users")
            
            # Start transaction for bulk insert
            db_conn.execute("BEGIN TRANSACTION")
            
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
                cur.execute("""
                    INSERT OR REPLACE INTO users 
                    (id, name, real_name, display_name, raw_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    user_id,
                    name,
                    real_name,
                    display_name,
                    json.dumps(user, separators=(",", ":"))
                ))
                
                users_processed += 1
            
            db_conn.commit()
            retry_count = 0  # Reset retry count on success
            
            # Check for more pages
            marker = j.get("next_marker")
            if not marker:
                break
                
            time.sleep(rate_limit)
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to fetch users after {max_retries} retries: {str(e)}")
                db_conn.rollback()
                raise
            
            logging.warning(f"Error fetching users (attempt {retry_count}/{max_retries}): {str(e)}")
            db_conn.rollback()
            time.sleep(rate_limit * 2)  # Exponential backoff
    
    if verbose:
        logging.info(f"Processed {users_processed} users")
    
    return users_processed


def get_user_display_name(db_conn, user_id):
    """Get the best available display name for a user."""
    if not user_id:
        return "Unknown"
        
    cur = db_conn.cursor()
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


def export_for_axolotl(db_conn, output_dir):
    """Export the database contents into a format suitable for Axolotl fine-tuning."""
    import os
    import json
    from datetime import datetime
    from collections import defaultdict
    import tempfile
    import shutil
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cur = db_conn.cursor()
    
    # Get all channels
    cur.execute("SELECT id, name FROM channels")
    channels = cur.fetchall()
    
    total_conversations = 0
    max_conversation_size = 100  # Maximum messages per conversation
    
    # Create a dataset file for each channel
    for channel_id, channel_name in channels:
        try:
            # Get all messages for this channel, including thread information
            cur.execute("""
                SELECT 
                    m.raw_json,
                    m.ts,
                    m.thread_ts,
                    CASE 
                        WHEN m.thread_ts IS NULL THEN m.ts 
                        ELSE m.thread_ts 
                    END as conversation_id
                FROM messages m
                WHERE m.channel_id = ?
                ORDER BY conversation_id, m.ts ASC
            """, (channel_id,))
            
            # Group messages by conversation (either thread or time-based)
            conversations = defaultdict(list)
            current_conversation = []
            last_ts = None
            last_conversation_id = None
            
            for raw_json, ts, thread_ts, conversation_id in cur.fetchall():
                try:
                    msg = json.loads(raw_json)
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON in message {ts} from channel {channel_id}")
                    continue
                
                # Skip messages without text
                if not msg.get('text'):
                    continue
                
                try:
                    # Get message timestamp
                    ts_float = float(ts)
                except (ValueError, TypeError):
                    logging.warning(f"Invalid timestamp {ts} in message from channel {channel_id}")
                    continue
                
                # Start a new conversation if:
                # 1. This is the first message
                # 2. There's a significant time gap (> 2 hours)
                # 3. The conversation ID has changed
                # 4. Current conversation is too large
                if (last_ts is None or 
                    ts_float - last_ts > 7200 or  # 2 hour gap
                    conversation_id != last_conversation_id or
                    len(current_conversation) >= max_conversation_size):
                    
                    # Save previous conversation if it exists and has multiple messages
                    if len(current_conversation) > 1:
                        conversations[last_conversation_id] = current_conversation
                    
                    # Start new conversation
                    current_conversation = []
                
                # Get user display name
                user_id = msg.get('user')
                user_name = get_user_display_name(db_conn, user_id) if user_id else "Slack"
                
                # Add message to current conversation
                current_conversation.append({
                    "role": "user" if user_id else "assistant",
                    "content": msg['text'],
                    "ts": ts,
                    "thread_ts": thread_ts,
                    "user": user_name,
                    "reactions": msg.get('reactions', []),
                    "is_thread_parent": thread_ts is None and msg.get('thread_ts') is not None
                })
                
                last_ts = ts_float
                last_conversation_id = conversation_id
            
            # Add the last conversation if it has multiple messages
            if len(current_conversation) > 1:
                conversations[last_conversation_id] = current_conversation
            
            # Convert conversations to Axolotl format
            axolotl_conversations = []
            for conv_id, messages in conversations.items():
                # Skip conversations with only one message
                if len(messages) < 2:
                    continue
                    
                # Format conversation for Axolotl
                conversation = {
                    "messages": [
                        {
                            "role": msg["role"],
                            "content": f"{msg['user']}: {msg['content']}"
                        }
                        for msg in messages
                    ],
                    "metadata": {
                        "channel": channel_name,
                        "channel_id": channel_id,
                        "conversation_id": conv_id,
                        "message_count": len(messages),
                        "is_thread": messages[0]["thread_ts"] is not None,
                        "participants": len(set(msg["user"] for msg in messages)),
                        "has_reactions": any(msg["reactions"] for msg in messages)
                    }
                }
                axolotl_conversations.append(conversation)
            
            # Skip channels with no conversations
            if not axolotl_conversations:
                continue
            
            # Create the dataset file
            dataset = {
                "type": "conversation",
                "conversations": axolotl_conversations,
                "channel": channel_name,
                "channel_id": channel_id,
                "stats": {
                    "total_conversations": len(axolotl_conversations),
                    "total_messages": sum(len(conv["messages"]) for conv in axolotl_conversations),
                    "avg_messages_per_conversation": sum(len(conv["messages"]) for conv in axolotl_conversations) / len(axolotl_conversations),
                    "thread_count": sum(1 for conv in axolotl_conversations if conv["metadata"]["is_thread"])
                }
            }
            
            # Write to temporary file first
            safe_name = "".join(c if c.isalnum() else "_" for c in channel_name)
            filename = f"{safe_name}_{channel_id}.json"
            output_path = os.path.join(output_dir, filename)
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                json.dump(dataset, temp_file, indent=2)
                temp_path = temp_file.name
            
            # Atomic move to final location
            shutil.move(temp_path, output_path)
            
            total_conversations += len(axolotl_conversations)
            print(f"Exported {len(axolotl_conversations)} conversations from #{channel_name} to {output_path}")
            print(f"  - Average messages per conversation: {dataset['stats']['avg_messages_per_conversation']:.1f}")
            print(f"  - Thread conversations: {dataset['stats']['thread_count']}")
            
        except Exception as e:
            logging.error(f"Error processing channel {channel_name}: {str(e)}")
            continue
    
    try:
        # Create metadata file
        metadata = {
            "export_date": datetime.now().isoformat(),
            "total_conversations": total_conversations,
            "channels": len(channels),
            "format": "axolotl-conversation",
            "version": "1.0",
            "channel_list": [name for _, name in channels]
        }
        
        # Create dataset config file for Axolotl
        dataset_config = {
            "datasets": [
                {
                    "path": os.path.join(output_dir, f"{safe_name}_{channel_id}.json"),
                    "type": "conversation"
                }
                for channel_id, channel_name in channels
            ]
        }
        
        # Write metadata files atomically
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(metadata, temp_file, indent=2)
            temp_path = temp_file.name
        shutil.move(temp_path, os.path.join(output_dir, "metadata.json"))
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(dataset_config, temp_file, indent=2)
            temp_path = temp_file.name
        shutil.move(temp_path, os.path.join(output_dir, "axolotl_config.json"))
        
        print(f"\nExported {total_conversations} total conversations from {len(channels)} channels")
        print(f"To train on all channels, use the generated axolotl_config.json")
        
    except Exception as e:
        logging.error(f"Error writing metadata files: {str(e)}")
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
    p.add_argument(
        "--dump-axolotl",
        help="Export database contents for Axolotl fine-tuning to the specified directory"
    )
    args = p.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")

    # DB
    conn = setup_db()

    # If dump-axolotl is specified, export the data and exit
    if args.dump_axolotl:
        export_for_axolotl(conn, args.dump_axolotl)
        conn.close()
        return

    base_url = f"https://{args.subdomain}.slack.com"
    # build session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; SlackArchiver/1.0)",
        "Cookie": args.cookie,
    })

    # Get internal token
    token = extract_token(session, base_url, args.verbose)

    # Fetch all users first
    fetch_all_users(session, args.org, token, args.rate_limit, args.verbose, conn)

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
