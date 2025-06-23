"""
Simplified Slack Data Exporter for LLM Fine-Tuning

This exporter creates high-quality Q&A training data from Slack conversations.

Format:
Each line in the JSONL output contains:
{
  "text": "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
}

Usage:
    from app import DatabaseManager
    from exporter import Exporter
    
    db = DatabaseManager()
    db.connect()
    exporter = Exporter(db)
    exporter.export_fine_tuning_data("./output")
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class Exporter:
    """Exports high-quality Q&A pairs from Slack for LLM fine-tuning."""

    def __init__(self, db):
        self.db = db

    def export_fine_tuning_data(self, output_dir: str, max_pairs_per_channel: int = 100) -> None:
        """Export high-quality Q&A pairs from Slack conversations.
        
        Args:
            output_dir: Directory to save the JSONL file
            max_pairs_per_channel: Maximum Q&A pairs to extract per channel
        """
        os.makedirs(output_dir, exist_ok=True)
        
        channels = self._get_all_channels()
        output_path = os.path.join(output_dir, "slack_qa_training.jsonl")
        
        total_pairs = 0
        channel_stats = defaultdict(int)
        
        print(f"\nExporting Q&A training data from {len(channels)} channels...")
        print(f"Maximum {max_pairs_per_channel} pairs per channel\n")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for channel_id, channel_name in channels:
                try:
                    # Extract Q&A pairs from this channel
                    qa_pairs = self._extract_channel_qa_pairs(channel_id, channel_name, max_pairs_per_channel)
                    
                    for qa_pair in qa_pairs:
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                        channel_stats[channel_name] += 1
                        total_pairs += 1
                    
                    if channel_stats[channel_name] > 0:
                        print(f"  #{channel_name}: {channel_stats[channel_name]} Q&A pairs")
                        
                except Exception as e:
                    logging.error(f"Error processing channel {channel_name}: {str(e)}")
                    continue
        
        # Write metadata
        self._write_export_metadata(output_dir, total_pairs, channel_stats)
        
        print(f"\nExport complete!")
        print(f"Total Q&A pairs: {total_pairs:,}")
        print(f"Output file: {output_path}")

    def _get_all_channels(self) -> List[Tuple[str, str]]:
        """Get all channels from the database."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT id, name FROM channels ORDER BY name")
        return cur.fetchall()

    def _extract_channel_qa_pairs(self, channel_id: str, channel_name: str, max_pairs: int) -> List[Dict]:
        """Extract high-quality Q&A pairs from a channel."""
        qa_pairs = []
        cur = self.db.conn.cursor()
        
        # Query threads with good Q&A potential (2-10 replies)
        cur.execute(
            """
            SELECT m.ts, m.user_id, m.reply_count
            FROM messages m
            WHERE m.channel_id = ?
            AND m.reply_count BETWEEN 2 AND 10
            AND (
                m.subtype is NULL
                OR m.subtype NOT IN ('channel_join', 'channel_leave', 'bot_message')
            )
            ORDER BY m.reply_count DESC, m.ts DESC
            LIMIT ?
            """,
            (channel_id, max_pairs * 2),  # Get extra for filtering
        )
        
        for row in cur.fetchall():
            if len(qa_pairs) >= max_pairs:
                break
                
            thread_ts, user_id, reply_count = row
            
            # Get the question (parent message)
            question = self._get_message_text(channel_id, thread_ts)
            if not question or not self._is_good_question(question):
                continue
            
            # Get the answer (best reply from thread)
            answer = self._get_best_thread_answer(channel_id, thread_ts)
            if not answer or len(answer) < 50:
                continue
            
            # Create training example
            qa_pairs.append({
                "text": f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            })
        
        # Also look for high-value explanatory messages
        if len(qa_pairs) < max_pairs:
            cur.execute(
                """
                SELECT m.ts, m.user_id
                FROM messages m  
                WHERE m.channel_id = ?
                AND m.thread_ts IS NULL
                AND (
                    m.subtype is NULL
                    OR m.subtype NOT IN ('channel_join', 'channel_leave', 'bot_message')
                )
                ORDER BY m.ts DESC
                LIMIT 500
                """,
                (channel_id,),
            )
            
            for row in cur.fetchall():
                if len(qa_pairs) >= max_pairs:
                    break
                    
                ts, user_id = row
                text = self._get_message_text(channel_id, ts)
                
                if text and self._is_valuable_explanation(text):
                    # Create a question for this explanation
                    topic = self._extract_topic(text)
                    if topic:
                        qa_pairs.append({
                            "text": f"<|im_start|>user\nCan you explain {topic}?<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>"
                        })
        
        return qa_pairs

    def _get_message_text(self, channel_id: str, ts: str) -> Optional[str]:
        """Get cleaned text content of a message."""
        raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
        if not raw_data:
            return None
            
        try:
            msg_data = json.loads(raw_data)
            return self._clean_slack_formatting(msg_data.get("text", ""))
        except (json.JSONDecodeError, KeyError):
            return None

    def _get_thread_messages(self, channel_id: str, thread_ts: str) -> List[Dict]:
        """Get all messages in a thread."""
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT m.ts
            FROM messages m
            WHERE m.channel_id = ? AND m.thread_ts = ?
            ORDER BY m.ts
            """,
            (channel_id, thread_ts),
        )
        
        thread_messages = []
        for row in cur.fetchall():
            ts = row[0]
            raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
            if raw_data:
                try:
                    thread_messages.append(json.loads(raw_data))
                except json.JSONDecodeError:
                    continue
                    
        return thread_messages

    def _get_best_thread_answer(self, channel_id: str, thread_ts: str) -> Optional[str]:
        """Extract the best answer from a thread."""
        thread_messages = self._get_thread_messages(channel_id, thread_ts)
        if len(thread_messages) < 2:
            return None
        
        best_answer = None
        best_score = 0
        
        # Skip the first message (the question)
        for msg in thread_messages[1:]:
            text = self._clean_slack_formatting(msg.get("text", ""))
            
            if len(text) < 30:
                continue
            
            # Score the answer quality
            score = self._score_answer_quality(text, msg)
            
            if score > best_score:
                best_score = score
                best_answer = text
        
        return best_answer

    def _score_answer_quality(self, text: str, msg_data: Dict) -> float:
        """Score the quality of an answer."""
        score = 0.0
        
        # Length bonus (substantial answers are better)
        if 50 < len(text) < 1000:
            score += 2.0
        elif 1000 <= len(text) < 2000:
            score += 1.5
        
        # Code blocks are valuable
        if '```' in text:
            score += 3.0
        elif text.count('`') >= 2:
            score += 1.5
        
        # Structured content (lists, steps)
        if any(marker in text for marker in ['1.', '2.', '- ', '* ', '•']):
            score += 2.0
        
        # Links often indicate references/solutions
        if 'http' in text:
            score += 1.0
        
        # Reactions indicate community validation
        reactions = msg_data.get("reactions", [])
        if reactions:
            score += min(sum(r.get("count", 0) for r in reactions) * 0.5, 3.0)
        
        # Files might contain important information
        if msg_data.get("files"):
            score += 1.5
        
        return score

    def _is_good_question(self, text: str) -> bool:
        """Check if text is a good question for training."""
        text = text.strip()
        
        # Too short or too long
        if len(text) < 15 or len(text) > 500:
            return False
        
        # Should be a question or problem statement
        text_lower = text.lower()
        
        # Direct questions
        if text.endswith('?'):
            return True
        
        # Question patterns
        question_patterns = [
            'how do', 'how can', 'how to', 'what is', 'what are', 'where',
            'when', 'why', 'does anyone', 'can someone', 'is there',
            'need help', 'looking for', 'trying to', 'want to', 'need to',
            'having trouble', 'getting error', 'issue with', 'problem with'
        ]
        
        return any(pattern in text_lower for pattern in question_patterns)

    def _is_valuable_explanation(self, text: str) -> bool:
        """Check if a message contains a valuable explanation."""
        # Must be substantial
        if len(text) < 200:
            return False
        
        text_lower = text.lower()
        
        # Look for explanatory patterns
        explanation_indicators = [
            'this is how', 'the way to', 'you can', 'you need to',
            'in order to', 'the process', 'here\'s how', 'solution is',
            'it works by', 'basically', 'essentially', 'the reason'
        ]
        
        has_explanation = any(indicator in text_lower for indicator in explanation_indicators)
        
        # Must also have substance (code, lists, or technical content)
        has_code = '```' in text or text.count('`') >= 2
        has_structure = any(marker in text for marker in ['1.', '2.', '- ', '* '])
        has_technical = self._has_technical_content(text_lower)
        
        return has_explanation and (has_code or has_structure or has_technical)

    def _has_technical_content(self, text_lower: str) -> bool:
        """Check if text contains technical content."""
        technical_terms = [
            'api', 'database', 'function', 'method', 'class', 'endpoint',
            'deployment', 'configuration', 'authentication', 'integration',
            'algorithm', 'architecture', 'implementation', 'optimization',
            'debug', 'error', 'exception', 'framework', 'library'
        ]
        
        matches = sum(1 for term in technical_terms if term in text_lower)
        return matches >= 2

    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract a topic from explanatory text."""
        # Try to get the first sentence or main subject
        first_sentence = text.split('.')[0].strip()
        
        # Clean common prefixes
        prefixes_to_remove = [
            'this is how', 'the way to', 'to', 'you can', 'you need to',
            'basically', 'essentially', 'so', 'well', 'ok', 'okay'
        ]
        
        topic = first_sentence.lower()
        for prefix in prefixes_to_remove:
            if topic.startswith(prefix + ' '):
                topic = topic[len(prefix):].strip()
        
        # Capitalize and clean
        topic = topic.strip(' ,.!?')
        if len(topic) > 10 and len(topic) < 100:
            return topic
        
        # Fallback: look for key technical terms
        text_lower = text.lower()
        for term in ['api', 'database', 'authentication', 'deployment', 'integration', 'configuration']:
            if term in text_lower:
                return term
        
        return None

    def _clean_slack_formatting(self, text: str) -> str:
        """Clean Slack-specific formatting."""
        if not text:
            return ""

        # Convert user mentions
        text = re.sub(r'<@([A-Z0-9]+)>', lambda m: f"@{self.db.get_user_display_name(m.group(1))}", text)

        # Convert channel mentions
        text = re.sub(r'<#([A-Z0-9]+)\|([^>]+)>', r'#\2', text)

        # Convert links
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _write_export_metadata(self, output_dir: str, total_pairs: int, channel_stats: Dict[str, int]) -> None:
        """Write metadata about the export."""
        metadata = {
            "export_type": "fine_tuning_qa",
            "exported_at": datetime.now().isoformat(),
            "format": {
                "type": "jsonl",
                "schema": {"text": "string containing formatted Q&A pair"},
                "template": "<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>"
            },
            "statistics": {
                "total_qa_pairs": total_pairs,
                "channels_processed": len(channel_stats),
                "pairs_per_channel": dict(sorted(channel_stats.items(), key=lambda x: x[1], reverse=True))
            },
            "quality_criteria": [
                "Questions must be clear and substantial (15-500 chars)",
                "Answers must be at least 50 characters",
                "Prioritizes threads with 2-10 replies",
                "Scores answers based on length, code content, structure, and community validation",
                "Includes high-value explanatory messages as Q&A pairs"
            ]
        }
        
        with open(os.path.join(output_dir, "export_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False) 