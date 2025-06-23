"""
Slack Q&A Exporter with optional LLM enhancement.

Usage:
    from app import DatabaseManager
    from exporter import Exporter
    
    db = DatabaseManager()
    db.connect()
    
    # Basic export
    exporter = Exporter(db)
    exporter.export_fine_tuning_data("./output")
    
    # Enhanced export with LLM
    from exporter import EnhancedExporter
    exporter = EnhancedExporter(db, model_name="microsoft/Phi-3-mini-4k-instruct")
    exporter.export_fine_tuning_data("./output")
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    answer: str
    channel: str
    quality_score: float = 0.0
    metadata: Dict = None
    
    def to_training_format(self) -> str:
        return f"<|im_start|>user\n{self.question}<|im_end|>\n<|im_start|>assistant\n{self.answer}<|im_end|>"
    
    def to_dict(self) -> Dict:
        result = {"text": self.to_training_format()}
        if self.metadata:
            result.update(self.metadata)
        return result


class BaseExporter:
    """Base exporter functionality shared by all exporters."""

    def __init__(self, db):
        self.db = db

    def export_fine_tuning_data(self, output_dir: str, max_pairs_per_channel: int = 100) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        channels = self._get_all_channels()
        output_path = os.path.join(output_dir, self._get_output_filename())
        
        total_pairs = 0
        channel_stats = defaultdict(int)
        
        print(f"\nExporting Q&A training data from {len(channels)} channels...")
        self._print_export_info(max_pairs_per_channel)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for channel_id, channel_name in channels:
                try:
                    qa_pairs = self._extract_channel_qa_pairs(channel_id, channel_name, max_pairs_per_channel)
                    
                    for qa_pair in qa_pairs:
                        f.write(json.dumps(qa_pair.to_dict(), ensure_ascii=False) + "\n")
                        channel_stats[channel_name] += 1
                        total_pairs += 1
                    
                    if channel_stats[channel_name] > 0:
                        print(f"  #{channel_name}: {channel_stats[channel_name]} Q&A pairs")
                        
                except Exception as e:
                    logging.error(f"Error processing channel {channel_name}: {str(e)}")
                    continue
        
        self._write_export_metadata(output_dir, total_pairs, channel_stats)
        
        print(f"\nExport complete!")
        print(f"Total Q&A pairs: {total_pairs:,}")
        print(f"Output file: {output_path}")
    
    def _get_output_filename(self) -> str:
        return "slack_qa_training.jsonl"
    
    def _print_export_info(self, max_pairs_per_channel: int = 100) -> None:
        print(f"Maximum pairs per channel: {max_pairs_per_channel}")

    def _get_all_channels(self) -> List[Tuple[str, str]]:
        cur = self.db.conn.cursor()
        cur.execute("SELECT id, name FROM channels ORDER BY name")
        return cur.fetchall()

    def _extract_channel_qa_pairs(self, channel_id: str, channel_name: str, max_pairs: int) -> List[QAPair]:
        qa_pairs = []
        
        # Extract from threads
        thread_pairs = self._extract_thread_qa_pairs(channel_id, channel_name, max_pairs)
        qa_pairs.extend(thread_pairs)
        
        # Extract from explanatory messages if needed
        if len(qa_pairs) < max_pairs:
            explanation_pairs = self._extract_explanation_pairs(channel_id, channel_name, max_pairs - len(qa_pairs))
            qa_pairs.extend(explanation_pairs)
        
        return qa_pairs[:max_pairs]

    def _extract_thread_qa_pairs(self, channel_id: str, channel_name: str, max_pairs: int) -> List[QAPair]:
        qa_pairs = []
        cur = self.db.conn.cursor()
        
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
            (channel_id, max_pairs * 2),
        )
        
        for row in cur.fetchall():
            thread_ts, user_id, reply_count = row
            
            question = self._get_message_text(channel_id, thread_ts)
            if not question or not self._is_good_question(question):
                continue
            
            answer = self._get_best_thread_answer(channel_id, thread_ts)
            if not answer or len(answer) < 50:
                continue
            
            qa_pair = self._create_qa_pair(question, answer, channel_name)
            if qa_pair:
                qa_pairs.append(qa_pair)
        
        return qa_pairs

    def _extract_explanation_pairs(self, channel_id: str, channel_name: str, max_pairs: int) -> List[QAPair]:
        qa_pairs = []
        cur = self.db.conn.cursor()
        
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
            ts, user_id = row
            text = self._get_message_text(channel_id, ts)
            
            if text and self._is_valuable_explanation(text):
                qa_pair = self._create_explanation_qa(text, channel_name)
                if qa_pair:
                    qa_pairs.append(qa_pair)
                    if len(qa_pairs) >= max_pairs:
                        break
        
        return qa_pairs

    def _create_qa_pair(self, question: str, answer: str, channel: str) -> Optional[QAPair]:
        """Create a QA pair with optional processing."""
        return QAPair(
            question=question,
            answer=answer,
            channel=channel,
            quality_score=self._score_qa_quality(question, answer)
        )

    def _create_explanation_qa(self, text: str, channel: str) -> Optional[QAPair]:
        """Create QA from an explanatory message."""
        topic = self._extract_topic(text)
        if topic:
            question = f"Can you explain {topic}?"
            return QAPair(
                question=question,
                answer=text,
                channel=channel,
                quality_score=self._score_explanation_quality(text)
            )
        return None

    def _get_message_text(self, channel_id: str, ts: str) -> Optional[str]:
        raw_data = self.db.get_compressed_data("messages", f"{channel_id}_{ts}")
        if not raw_data:
            return None
            
        try:
            msg_data = json.loads(raw_data)
            return self._clean_slack_formatting(msg_data.get("text", ""))
        except (json.JSONDecodeError, KeyError):
            return None

    def _get_thread_messages(self, channel_id: str, thread_ts: str) -> List[Dict]:
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
        thread_messages = self._get_thread_messages(channel_id, thread_ts)
        if len(thread_messages) < 2:
            return None
        
        best_answer = None
        best_score = 0
        
        for msg in thread_messages[1:]:
            text = self._clean_slack_formatting(msg.get("text", ""))
            
            if len(text) < 30:
                continue
            
            score = self._score_answer_quality(text, msg)
            
            if score > best_score:
                best_score = score
                best_answer = text
        
        return best_answer

    def _score_answer_quality(self, text: str, msg_data: Dict) -> float:
        score = 0.0
        
        if 50 < len(text) < 1000:
            score += 2.0
        elif 1000 <= len(text) < 2000:
            score += 1.5
        
        if '```' in text:
            score += 3.0
        elif text.count('`') >= 2:
            score += 1.5
        
        if any(marker in text for marker in ['1.', '2.', '- ', '* ', '•']):
            score += 2.0
        
        if 'http' in text:
            score += 1.0
        
        reactions = msg_data.get("reactions", [])
        if reactions:
            score += min(sum(r.get("count", 0) for r in reactions) * 0.5, 3.0)
        
        if msg_data.get("files"):
            score += 1.5
        
        return score

    def _score_qa_quality(self, question: str, answer: str) -> float:
        """Basic quality scoring for Q&A pairs."""
        score = 5.0
        
        if question.endswith('?'):
            score += 0.5
        if len(question) > 20:
            score += 0.5
        if len(answer) > 100:
            score += 1.0
        if '```' in answer:
            score += 1.0
        if any(marker in answer for marker in ['1.', '2.', '- ']):
            score += 0.5
            
        return min(score, 10.0)

    def _score_explanation_quality(self, text: str) -> float:
        """Score quality of explanatory text."""
        return self._score_qa_quality("", text)

    def _is_good_question(self, text: str) -> bool:
        text = text.strip()
        
        if len(text) < 15 or len(text) > 500:
            return False
        
        text_lower = text.lower()
        
        if text.endswith('?'):
            return True
        
        question_patterns = [
            'how do', 'how can', 'how to', 'what is', 'what are', 'where',
            'when', 'why', 'does anyone', 'can someone', 'is there',
            'need help', 'looking for', 'trying to', 'want to', 'need to',
            'having trouble', 'getting error', 'issue with', 'problem with'
        ]
        
        return any(pattern in text_lower for pattern in question_patterns)

    def _is_valuable_explanation(self, text: str) -> bool:
        if len(text) < 200:
            return False
        
        text_lower = text.lower()
        
        explanation_indicators = [
            'this is how', 'the way to', 'you can', 'you need to',
            'in order to', 'the process', 'here\'s how', 'solution is',
            'it works by', 'basically', 'essentially', 'the reason'
        ]
        
        has_explanation = any(indicator in text_lower for indicator in explanation_indicators)
        
        has_code = '```' in text or text.count('`') >= 2
        has_structure = any(marker in text for marker in ['1.', '2.', '- ', '* '])
        has_technical = self._has_technical_content(text_lower)
        
        return has_explanation and (has_code or has_structure or has_technical)

    def _has_technical_content(self, text_lower: str) -> bool:
        technical_terms = [
            'api', 'database', 'function', 'method', 'class', 'endpoint',
            'deployment', 'configuration', 'authentication', 'integration',
            'algorithm', 'architecture', 'implementation', 'optimization',
            'debug', 'error', 'exception', 'framework', 'library'
        ]
        
        matches = sum(1 for term in technical_terms if term in text_lower)
        return matches >= 2

    def _extract_topic(self, text: str) -> Optional[str]:
        first_sentence = text.split('.')[0].strip()
        
        prefixes_to_remove = [
            'this is how', 'the way to', 'to', 'you can', 'you need to',
            'basically', 'essentially', 'so', 'well', 'ok', 'okay'
        ]
        
        topic = first_sentence.lower()
        for prefix in prefixes_to_remove:
            if topic.startswith(prefix + ' '):
                topic = topic[len(prefix):].strip()
        
        topic = topic.strip(' ,.!?')
        if len(topic) > 10 and len(topic) < 100:
            return topic
        
        text_lower = text.lower()
        for term in ['api', 'database', 'authentication', 'deployment', 'integration', 'configuration']:
            if term in text_lower:
                return term
        
        return None

    def _clean_slack_formatting(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'<@([A-Z0-9]+)>', lambda m: f"@{self.db.get_user_display_name(m.group(1))}", text)
        text = re.sub(r'<#([A-Z0-9]+)\|([^>]+)>', r'#\2', text)
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _write_export_metadata(self, output_dir: str, total_pairs: int, channel_stats: Dict[str, int]) -> None:
        metadata = {
            "export_type": self._get_export_type(),
            "exported_at": datetime.now().isoformat(),
            "format": {
                "type": "jsonl",
                "schema": {"text": "formatted Q&A pair"},
                "template": "<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>"
            },
            "statistics": {
                "total_qa_pairs": total_pairs,
                "channels_processed": len(channel_stats),
                "pairs_per_channel": dict(sorted(channel_stats.items(), key=lambda x: x[1], reverse=True))
            }
        }
        
        metadata.update(self._get_additional_metadata())
        
        with open(os.path.join(output_dir, "export_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _get_export_type(self) -> str:
        return "basic_qa"
    
    def _get_additional_metadata(self) -> Dict:
        return {}


class Exporter(BaseExporter):
    """Standard exporter without LLM enhancement."""
    pass


class EnhancedExporter(BaseExporter):
    """Exporter with LLM enhancement capabilities."""
    
    def __init__(self, db, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        super().__init__(db)
        self.model_name = model_name
        self.llm = None
        self.enhanced_count = 0
        self.synthetic_count = 0
        
    def _print_export_info(self, max_pairs_per_channel: int = 100) -> None:
        super()._print_export_info(max_pairs_per_channel)
        print(f"Using LLM enhancement: {self.model_name}")
        
    def _get_output_filename(self) -> str:
        return "enhanced_qa_training.jsonl"
    
    def _get_export_type(self) -> str:
        return "llm_enhanced_qa"
    
    def _get_additional_metadata(self) -> Dict:
        return {
            "enhancements": {
                "model": self.model_name,
                "enhanced_answers": self.enhanced_count,
                "synthetic_pairs": self.synthetic_count
            }
        }

    def _load_llm(self):
        """Lazy load LLM when needed."""
        if self.llm is not None:
            return
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading LLM: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            self.llm = True
        except ImportError:
            print("Warning: transformers not installed, falling back to basic export")
            self.llm = False
    
    def _create_qa_pair(self, question: str, answer: str, channel: str) -> Optional[QAPair]:
        """Create enhanced QA pair."""
        self._load_llm()
        
        if self.llm:
            enhanced_answer = self._enhance_answer(question, answer, channel)
            if enhanced_answer != answer:
                self.enhanced_count += 1
                answer = enhanced_answer
        
        return QAPair(
            question=question,
            answer=answer,
            channel=channel,
            quality_score=self._score_qa_quality(question, answer),
            metadata={"enhanced": self.llm and enhanced_answer != answer}
        )
    
    def _create_explanation_qa(self, text: str, channel: str) -> Optional[QAPair]:
        """Create synthetic Q&A from explanation."""
        self._load_llm()
        
        if self.llm:
            synthetic_pairs = self._generate_questions_for_explanation(text, channel)
            if synthetic_pairs:
                self.synthetic_count += len(synthetic_pairs)
                return synthetic_pairs[0]  # Return best one
        
        # Fallback to basic extraction
        return super()._create_explanation_qa(text, channel)
    
    def _enhance_answer(self, question: str, answer: str, channel: str) -> str:
        """Enhance answer using LLM."""
        if len(answer) > 500 and '```' in answer:
            return answer  # Already high quality
            
        prompt = f"""Improve this answer to be more complete and helpful.

Question: {question}
Answer: {answer}

Enhanced answer:"""
        
        try:
            enhanced = self._generate(prompt, max_tokens=600)
            
            # Quality check
            if 0.5 < len(enhanced) / len(answer) < 3.0:
                return enhanced
        except:
            pass
            
        return answer
    
    def _generate_questions_for_explanation(self, text: str, channel: str) -> List[QAPair]:
        """Generate questions that this text answers."""
        prompt = f"""This message contains useful information:

{text[:500]}

Generate 2 specific questions this answers. Format: Q1: ... Q2: ..."""
        
        try:
            response = self._generate(prompt, max_tokens=150)
            qa_pairs = []
            
            for line in response.split('\n'):
                if line.strip().startswith(('Q1:', 'Q2:')):
                    question = line.split(':', 1)[1].strip()
                    if len(question) > 15:
                        qa_pairs.append(QAPair(
                            question=question,
                            answer=text,
                            channel=channel,
                            quality_score=self._score_explanation_quality(text),
                            metadata={"synthetic": True}
                        ))
            
            return qa_pairs
        except:
            return []
    
    def _score_qa_quality(self, question: str, answer: str) -> float:
        """Enhanced scoring using LLM if available."""
        if not self.llm:
            return super()._score_qa_quality(question, answer)
            
        try:
            prompt = f"""Rate this Q&A quality 0-10:
Q: {question[:100]}
A: {answer[:200]}...

Just reply with a number:"""
            
            response = self._generate(prompt, max_tokens=10)
            
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                return min(float(numbers[0]), 10.0)
        except:
            pass
            
        return super()._score_qa_quality(question, answer)
    
    def _generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text using the LLM."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip() 