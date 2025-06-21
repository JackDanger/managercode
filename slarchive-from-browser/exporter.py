import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class Exporter:
    """Handles data export operations for LLM fine-tuning."""

    # Conservative defaults for basic export
    MAX_CONTEXT_MESSAGES = 10
    MAX_TOKENS_PER_CONVERSATION = 2048

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

    def __init__(self, db):
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

    def export_for_fine_tuning(self, output_dir: str) -> None:
        """Export Slack data for fine-tuning, focusing on complete threads and conversations."""
        os.makedirs(output_dir, exist_ok=True)
        
        channels = self._get_all_channels()
        output_path = os.path.join(output_dir, "slack_fine_tune.jsonl")
        
        total_conversations = 0
        total_messages = 0
        
        print(f"\nExporting {len(channels)} channels for fine-tuning to {output_dir}...")
        print("Bot messages are automatically filtered out at the database level")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for channel_id, channel_name in channels:
                try:
                    print(f"Processing channel #{channel_name}...")
                    channel_conversations = 0
                    
                    # Process all threads in the channel
                    conversations = self._process_channel_for_fine_tuning(channel_id, channel_name)
                    
                    for conv in conversations:
                        # Create training example
                        training_example = self._create_fine_tune_example(conv)
                        if training_example:
                            f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
                            channel_conversations += 1
                            total_messages += conv.get("message_count", 0)
                    
                    total_conversations += channel_conversations
                    print(f"  Exported {channel_conversations} conversations")
                    
                except Exception as e:
                    logging.error(f"Error processing channel {channel_name}: {str(e)}")
                    continue
        
        # Write metadata
        self._write_fine_tune_metadata(output_dir, total_conversations, total_messages, channels)
        
        print(f"\nFine-tune export complete:")
        print(f"  Total conversations: {total_conversations:,}")
        print(f"  Total messages: {total_messages:,}")
        print(f"Output: {output_path}")

    def _process_channel_for_fine_tuning(self, channel_id: str, channel_name: str) -> List[Dict]:
        """Process a channel to extract complete conversations for fine-tuning."""
        conversations = []
        cur = self.db.conn.cursor()
        
        # Get all non-bot messages ordered by timestamp
        cur.execute(
            """
            SELECT m.ts, m.user_id, m.thread_ts, m.subtype, m.reply_count
            FROM messages m
            WHERE m.channel_id = ?
            AND (
                   m.subtype is NULL
                OR m.subtype NOT IN ('channel_join', 'channel_leave', 'bot_message')
            )
            ORDER BY m.ts
        """,
            (channel_id,),
        )
        
        processed_threads = set()
        standalone_messages = []
        current_context = []
        last_ts = None
        
        for msg_row in cur.fetchall():
            ts, user_id, thread_ts, subtype, reply_count = msg_row
            
            # Get full message data
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if not raw_data:
                continue
                
            msg_data = json.loads(raw_data)
            
            # Handle threads
            if thread_ts and thread_ts not in processed_threads:
                # Skip - will process when we hit the parent
                continue
            elif not thread_ts and reply_count and reply_count > 0 and ts not in processed_threads:
                # This starts a thread - get all messages
                thread_conv = self._extract_thread_conversation(channel_id, channel_name, ts, msg_data)
                if thread_conv:
                    conversations.append(thread_conv)
                processed_threads.add(ts)
            elif not thread_ts:
                # Standalone message - collect for context windows
                user_name = self.db.get_user_display_name(user_id)
                formatted_msg = self._clean_slack_formatting(msg_data.get("text", ""))
                
                # Check if we should start a new context window (>30 minutes gap)
                if last_ts and float(ts) - float(last_ts) > 1800:  # 30 minutes
                    if current_context:
                        # Save previous context as a conversation
                        context_conv = self._create_context_conversation(channel_name, current_context)
                        if context_conv:
                            conversations.append(context_conv)
                    current_context = []
                
                current_context.append({
                    "ts": ts,
                    "user": user_name,
                    "text": formatted_msg,
                    "raw_data": msg_data
                })
                last_ts = ts
        
        # Don't forget the last context window
        if current_context:
            context_conv = self._create_context_conversation(channel_name, current_context)
            if context_conv:
                conversations.append(context_conv)
        
        return conversations

    def _extract_thread_conversation(self, channel_id: str, channel_name: str, thread_ts: str, parent_msg: Dict) -> Optional[Dict]:
        """Extract a complete thread as a conversation."""
        thread_messages = self._get_thread_messages(channel_id, thread_ts)
        
        # Combine parent and thread messages
        all_messages = [parent_msg] + thread_messages
        
        # Format messages
        formatted_messages = []
        
        for msg in all_messages:
            user_id = msg.get("user")
            if not user_id:
                continue
                
            user_name = self.db.get_user_display_name(user_id)
            text = self._clean_slack_formatting(msg.get("text", ""))
            
            if text:
                formatted_messages.append({
                    "user": user_name,
                    "text": text,
                    "ts": msg.get("ts")
                })
        
        # Skip if too few messages after formatting
        if len(formatted_messages) < 2:
            return None
            
        return {
            "type": "thread",
            "channel": channel_name,
            "messages": formatted_messages,
            "message_count": len(formatted_messages),
            "thread_ts": thread_ts
        }

    def _create_context_conversation(self, channel_name: str, messages: List[Dict]) -> Optional[Dict]:
        """Create a conversation from a context window of messages."""
        if len(messages) < 2:
            return None
            
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "user": msg["user"],
                "text": msg["text"],
                "ts": msg["ts"]
            })
            
        return {
            "type": "context_window",
            "channel": channel_name,
            "messages": formatted_messages,
            "message_count": len(formatted_messages)
        }

    def _create_fine_tune_example(self, conversation: Dict) -> Optional[Dict]:
        """Create a fine-tuning example from a conversation."""
        messages = conversation.get("messages", [])
        if len(messages) < 2:
            return None
            
        # Build the conversation text
        conversation_lines = []
        for msg in messages:
            conversation_lines.append(f"{msg['user']}: {msg['text']}")
        
        conversation_text = "\n".join(conversation_lines)
        
        # Create a training example that captures the conversational nature
        # We'll create a system prompt that explains this is a Slack conversation
        system_prompt = f"You are an AI assistant that has learned from the Slack conversations of an organization. You understand their technical discussions, processes, and collaborative problem-solving approaches. This conversation is from the #{conversation['channel']} channel."
        
        # Create a user prompt that asks about the conversation
        user_prompt = f"Here's a Slack conversation from our team:\n\n{conversation_text}\n\nBased on this discussion, what can you tell me about the topic being discussed and how the team approached solving it?"
        
        # Create an assistant response that demonstrates understanding
        assistant_response = "Based on this Slack conversation, I can analyze the discussion and explain the key points, problem-solving approach, and any conclusions reached. Let me break down what happened in this thread."
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ],
            "metadata": {
                "channel": conversation["channel"],
                "type": conversation["type"],
                "message_count": conversation["message_count"],
                "thread_ts": conversation.get("thread_ts")
            }
        }

    def _clean_slack_formatting(self, text: str) -> str:
        """Clean Slack-specific formatting for better training data."""
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

    def _write_fine_tune_metadata(self, output_dir: str, total_conversations: int, total_messages: int, channels: List[Tuple[str, str]]) -> None:
        """Write metadata for fine-tuning export."""
        metadata = {
            "export_type": "fine_tune",
            "exported_at": datetime.now().isoformat(),
            "statistics": {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "channels": len(channels)
            },
            "format": {
                "description": "Each line contains a training example with system/user/assistant messages",
                "conversation_types": [
                    "thread - Complete discussion threads with multiple participants",
                    "context_window - Related messages within 30-minute windows"
                ]
            },
            "training_notes": [
                "Optimized for capturing collaborative problem-solving patterns",
                "Preserves full context of technical discussions",
                "Bot messages are filtered out at the database level",
                "Suitable for fine-tuning models like Qwen3 or Mistral"
            ]
        }
        
        with open(os.path.join(output_dir, "fine_tune_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

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
        }

        print(f"\nStreaming export of {len(channels)} channels for RAG to {output_dir}...")
        print("Bot messages are automatically filtered out at the database level")

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
        print(f"Output: {output_path}")

    def _stream_channel_for_rag(self, channel_id: str, channel_name: str, include_thread_summaries: bool, batch_size: int, stats: Dict):
        """Stream RAG-optimized documents for a channel using keyset pagination for optimal performance."""
        cur = self.db.conn.cursor()
        processed_threads = set()
        last_ts = None  # Cursor for keyset pagination
        processed_count = 0

        while True:
            # Use keyset pagination with timestamp cursor for consistent performance
            if last_ts is None:
                # First batch - start from beginning
                cur.execute(
                    """
                    SELECT m.ts, m.user_id, m.thread_ts, m.subtype, m.reply_count
                    FROM messages m
                    WHERE m.channel_id = ?
                    AND (
                        m.subtype is NULL
                        OR m.subtype NOT IN ('channel_join', 'channel_leave', 'bot_message')
                    )
                    ORDER BY m.ts
                    LIMIT ?
                    """,
                    (channel_id, batch_size),
                )
            else:
                # Subsequent batches - continue from last timestamp
                cur.execute(
                    """
                    SELECT m.ts, m.user_id, m.thread_ts, m.subtype, m.reply_count
                    FROM messages m
                    WHERE m.channel_id = ?
                    AND (
                        m.subtype is NULL
                        OR m.subtype NOT IN ('channel_join', 'channel_leave', 'bot_message')
                    )
                    AND m.ts > ?
                    ORDER BY m.ts
                    LIMIT ?
                    """,
                    (channel_id, last_ts, batch_size),
                )

            batch = cur.fetchall()
            if not batch:
                break

            # Process batch and update cursor
            for msg_row in batch:
                ts, user_id, thread_ts, subtype, reply_count = msg_row
                last_ts = ts  # Update cursor to this timestamp

                # Get the full message data
                raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
                if not raw_data:
                    continue

                try:
                    msg_data = json.loads(raw_data)

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

                    processed_count += 1

                    # Progress update for large channels
                    if processed_count % (batch_size * 10) == 0:
                        print(f"    Processed {processed_count:,} messages...")

                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Error processing message {channel_id}_{ts}: {e}")
                    continue

        # Clean up
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