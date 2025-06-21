#!/usr/bin/env python3
"""
RAG Query Application for Slack Archive

This application loads documents exported by the Slack archiver and provides
a question-answering interface using Hugging Face models and semantic search.
"""

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
from tqdm import tqdm


@dataclass
class Document:
    """Represents a single document from the RAG export."""
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def __str__(self):
        channel = self.metadata.get('channel', 'unknown')
        user = self.metadata.get('user', 'unknown')
        date = self.metadata.get('date', 'unknown')
        return f"#{channel} | {user} | {date}: {self.content[:100]}..."


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    document: Document
    score: float
    rank: int


class DocumentStore:
    """Manages document storage and retrieval."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
    def load_documents(self, jsonl_path: str) -> None:
        """Load documents from the RAG export JSONL file."""
        print(f"Loading documents from {jsonl_path}...")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"RAG export file not found: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        id=data['id'],
                        content=data['content'],
                        metadata=data['metadata']
                    )
                    self.documents.append(doc)
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self.documents):,} documents")
        
    def build_index(self, cache_dir: str = "./cache") -> None:
        """Build or load the vector index for semantic search."""
        os.makedirs(cache_dir, exist_ok=True)
        
        embeddings_cache = os.path.join(cache_dir, f"embeddings_{self.embedding_model_name.replace('/', '_')}.pkl")
        index_cache = os.path.join(cache_dir, f"faiss_index_{self.embedding_model_name.replace('/', '_')}.index")
        
        # Try to load cached embeddings and index
        if os.path.exists(embeddings_cache) and os.path.exists(index_cache):
            print("Loading cached embeddings and index...")
            with open(embeddings_cache, 'rb') as f:
                self.embeddings = pickle.load(f)
            self.index = faiss.read_index(index_cache)
            print("Cached embeddings and index loaded successfully")
            return
        
        # Build new embeddings and index
        print(f"Building embeddings using {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Extract content for embedding
        texts = [doc.content for doc in self.documents]
        
        # Create embeddings in batches to manage memory
        batch_size = 32
        embeddings_list = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        # Cache embeddings and index
        print("Caching embeddings and index...")
        with open(embeddings_cache, 'wb') as f:
            pickle.dump(self.embeddings, f)
        faiss.write_index(self.index, index_cache)
        
        print(f"Index built with {self.index.ntotal:,} documents")
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[SearchResult]:
        """Search for relevant documents using semantic similarity."""
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if score >= min_score:  # Filter out low-relevance results
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank + 1
                ))
        
        return results
    
    def filter_documents(self, 
                        channel: Optional[str] = None,
                        user: Optional[str] = None,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None,
                        content_type: Optional[str] = None) -> List[Document]:
        """Filter documents by metadata criteria."""
        filtered = []
        
        for doc in self.documents:
            meta = doc.metadata
            
            # Channel filter
            if channel and meta.get('channel', '').lower() != channel.lower():
                continue
                
            # User filter  
            if user and user.lower() not in meta.get('user', '').lower():
                continue
                
            # Date range filter
            if date_from or date_to:
                doc_date = meta.get('date', '')
                if date_from and doc_date < date_from:
                    continue
                if date_to and doc_date > date_to:
                    continue
            
            # Content type filter
            if content_type and content_type not in meta.get('content_type', ''):
                continue
                
            filtered.append(doc)
        
        return filtered


class RAGQueryEngine:
    """Main query engine that combines document retrieval with language model generation."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 max_context_length: int = 2048,
                 max_new_tokens: int = 512):
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.document_store = DocumentStore()
        
    def load_model(self) -> None:
        """Load the Hugging Face model and tokenizer."""
        print(f"Loading model {self.model_name}...")
        
        try:
            # Try to use a text generation pipeline first (easier)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                max_length=self.max_context_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # Common pad token
            )
            print("Model loaded successfully using pipeline")
            
        except Exception as e:
            print(f"Pipeline loading failed: {e}")
            print("Falling back to direct model loading...")
            
            # Fallback to direct model loading
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("Model loaded successfully")
    
    def setup_documents(self, jsonl_path: str, cache_dir: str = "./cache") -> None:
        """Setup document store with RAG documents."""
        self.document_store.load_documents(jsonl_path)
        self.document_store.build_index(cache_dir)
    
    def _create_context(self, results: List[SearchResult], max_context_chars: int = 4000) -> str:
        """Create context string from search results."""
        context_parts = []
        current_length = 0
        
        for result in results:
            doc = result.document
            meta = doc.metadata
            
            # Format document with metadata
            doc_text = f"Channel: #{meta.get('channel', 'unknown')}\n"
            doc_text += f"User: {meta.get('user', 'unknown')}\n"
            doc_text += f"Date: {meta.get('date', 'unknown')}\n"
            doc_text += f"Content: {doc.content}\n"
            doc_text += f"Relevance: {result.score:.3f}\n"
            doc_text += "---\n"
            
            if current_length + len(doc_text) > max_context_chars:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def query(self, 
              question: str, 
              top_k: int = 5, 
              include_metadata: bool = True,
              filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        start_time = time.time()
        
        # Search for relevant documents
        search_results = self.document_store.search(question, top_k=top_k)
        
        if not search_results:
            return {
                "question": question,
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "search_time": time.time() - start_time,
                "generation_time": 0
            }
        
        # Create context from search results
        context = self._create_context(search_results)
        
        # Create the prompt
        prompt = f"""Based on the following Slack conversation history, please answer the question.

Context:
{context}

Question: {question}

Answer: """

        generation_start = time.time()
        
        # Generate answer
        if self.pipeline:
            # Use pipeline
            try:
                response = self.pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.pad_token_id or 50256
                )
                answer = response[0]['generated_text'][len(prompt):].strip()
            except Exception as e:
                answer = f"Error generating response: {str(e)}"
                
        else:
            # Use direct model
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_length)
                
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = full_response[len(prompt):].strip()
                
            except Exception as e:
                answer = f"Error generating response: {str(e)}"
        
        generation_time = time.time() - generation_start
        
        # Prepare sources
        sources = []
        if include_metadata:
            for result in search_results:
                doc = result.document
                sources.append({
                    "channel": doc.metadata.get('channel', 'unknown'),
                    "user": doc.metadata.get('user', 'unknown'),
                    "date": doc.metadata.get('date', 'unknown'),
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "relevance_score": result.score,
                    "document_id": doc.id
                })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "search_time": time.time() - start_time - generation_time,
            "generation_time": generation_time,
            "total_time": time.time() - start_time,
            "context_length": len(context),
            "prompt_length": len(prompt)
        }
    
    def interactive_query(self) -> None:
        """Run an interactive query session."""
        print("\n=== Slack Archive RAG Query System ===")
        print("Type 'quit' to exit, 'help' for commands")
        print(f"Loaded {len(self.document_store.documents):,} documents")
        print()
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                print("🔍 Searching...")
                result = self.query(question)
                
                print(f"\n💬 Answer:")
                print(result['answer'])
                
                print(f"\n📚 Sources ({len(result['sources'])} documents):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. #{source['channel']} | {source['user']} | {source['date']}")
                    print(f"     {source['content']}")
                    print(f"     (Relevance: {source['relevance_score']:.3f})")
                
                print(f"\n⏱️  Timing: Search: {result['search_time']:.2f}s, Generation: {result['generation_time']:.2f}s")
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def _show_help(self) -> None:
        """Show help information."""
        print("""
Available commands:
  help     - Show this help
  quit     - Exit the program
  
Example questions:
  - "How do we deploy to production?"
  - "What are the database migration steps?"
  - "Who worked on the authentication system?"
  - "What was discussed about the API rate limiting?"
  
Tips:
  - Be specific in your questions for better results
  - Questions about processes, decisions, and technical discussions work best
  - The system searches through Slack message content and thread discussions
        """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query Slack archive using RAG")
    
    parser.add_argument(
        "rag_file", 
        help="Path to the RAG export JSONL file (slack_rag_documents.jsonl)"
    )
    parser.add_argument(
        "--model", 
        default="microsoft/DialoGPT-medium",
        help="Hugging Face model to use for generation (default: microsoft/DialoGPT-medium)"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Directory to cache embeddings and indices (default: ./cache)"
    )
    parser.add_argument(
        "--query",
        help="Single query to execute (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve for context (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the query engine
    engine = RAGQueryEngine(
        model_name=args.model,
        max_context_length=2048,
        max_new_tokens=512
    )
    
    # Load model
    engine.load_model()
    
    # Setup document store
    engine.document_store = DocumentStore(embedding_model=args.embedding_model)
    engine.setup_documents(args.rag_file, args.cache_dir)
    
    if args.query:
        # Single query mode
        result = engine.query(args.query, top_k=args.top_k)
        
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. #{source['channel']} | {source['user']} | {source['date']}")
        print(f"\nTiming: {result['total_time']:.2f}s")
        
    else:
        # Interactive mode
        engine.interactive_query()


if __name__ == "__main__":
    main() 