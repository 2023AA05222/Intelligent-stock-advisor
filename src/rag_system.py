"""
Advanced RAG System for Financial Analysis
Implements HNSW indexing, Semantic Chunking, Hybrid Search, Query Rewriting, 
Autocut, Reranking, and RAFT techniques
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import json
import pickle
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Vector search and embeddings
try:
    import faiss
    import hnswlib
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some RAG dependencies not installed: {e}")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    # Create a dummy class for type hints when the package is not installed
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

# Embedding model configurations
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "MiniLM-L6-v2 (Fast)",
        "dimension": 384,
        "description": "Lightweight and fast, good for general purpose",
        "trust_remote_code": False
    },
    "nomic-ai/nomic-embed-text-v1": {
        "name": "Nomic Embed v1 (High Quality)",
        "dimension": 768,
        "description": "High quality embeddings, better semantic understanding",
        "trust_remote_code": True
    }
}

# Reranking and advanced retrieval
try:
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logging.warning(f"Some ML dependencies not installed: {e}")

# Text processing
import re
try:
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    # Fallback to basic sentence splitting if NLTK not available
    def sent_tokenize(text):
        return text.split('. ')
    def word_tokenize(text):
        return text.split()

try:
    import spacy
except ImportError:
    spacy = None

@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""
    source: str = ""
    timestamp: datetime = None
    semantic_score: float = 0.0

@dataclass
class QueryRewrite:
    """Represents a rewritten query with metadata"""
    original_query: str
    rewritten_queries: List[str]
    query_type: str  # "financial", "technical", "news", "company"
    intent: str
    entities: List[str]

class SemanticChunker:
    """Advanced semantic chunking using sentence embeddings and coherence scoring"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        model_config = EMBEDDING_MODELS.get(model_name, {"trust_remote_code": False})
        trust_remote_code = model_config.get("trust_remote_code", False)
        
        try:
            self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            # Try local cache paths
            import os
            cache_paths = [
                "/app/.cache/transformers",
                "/app/.cache/huggingface", 
                os.path.expanduser("~/.cache/huggingface"),
                "./models"
            ]
            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    try:
                        os.environ['TRANSFORMERS_CACHE'] = cache_path
                        os.environ['HF_HOME'] = cache_path
                        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
                        break
                    except Exception:
                        continue
            else:
                raise RuntimeError(f"Cannot load sentence transformer model: {e}")
        self.coherence_threshold = 0.7
        self.max_chunk_size = 512
        self.min_chunk_size = 50
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Create semantically coherent chunks from text"""
        # Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            # Single sentence - still need embedding
            embedding = self.model.encode([text])[0]
            return [SemanticChunk(content=text, metadata=metadata, embedding=embedding)]
        
        # Get sentence embeddings
        sentence_embeddings = self.model.encode(sentences)
        
        # Calculate semantic similarity between adjacent sentences
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = [sentence_embeddings[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            chunk_embedding = np.mean(current_embedding, axis=0)
            similarity = cosine_similarity([chunk_embedding], [sentence_embeddings[i]])[0][0]
            
            # Check chunk size
            current_text = " ".join(current_chunk + [sentences[i]])
            
            if (similarity >= self.coherence_threshold and 
                len(current_text.split()) <= self.max_chunk_size):
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_embedding.append(sentence_embeddings[i])
            else:
                # Create new chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunk_embedding = np.mean(current_embedding, axis=0)
                    chunks.append(SemanticChunk(
                        content=chunk_text,
                        metadata=metadata.copy(),
                        embedding=chunk_embedding,
                        semantic_score=similarity
                    ))
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_embedding = [sentence_embeddings[i]]
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunk_embedding = np.mean(current_embedding, axis=0)
                chunks.append(SemanticChunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    embedding=chunk_embedding
                ))
        
        return chunks

class HNSWVectorStore:
    """HNSW-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = 384, max_elements: int = 100000):
        self.dimension = dimension
        self.max_elements = max_elements
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=max_elements, ef_construction=400, M=32)
        self.index.set_ef(200)  # Set ef for search
        self.chunks: Dict[int, SemanticChunk] = {}
        self.next_id = 0
        self.is_initialized = True
    
    def reinitialize(self, dimension: int):
        """Reinitialize the vector store with new dimensions"""
        if dimension != self.dimension:
            self.dimension = dimension
            self.index = hnswlib.Index(space='cosine', dim=dimension)
            self.index.init_index(max_elements=self.max_elements, ef_construction=400, M=32)
            self.index.set_ef(200)
            self.chunks = {}
            self.next_id = 0
            self.is_initialized = True
        
    def add_chunks(self, chunks: List[SemanticChunk]):
        """Add chunks to the vector store"""
        embeddings = []
        ids = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                chunk.chunk_id = str(self.next_id)
                self.chunks[self.next_id] = chunk
                embeddings.append(chunk.embedding)
                ids.append(self.next_id)
                self.next_id += 1
        
        if embeddings:
            self.index.add_items(np.array(embeddings), ids)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[SemanticChunk, float]]:
        """Search for similar chunks"""
        if self.next_id == 0:
            return []
        
        try:
            # Ensure we don't search for more items than we have
            actual_k = min(k, self.next_id)
            self.index.set_ef(max(actual_k * 2, 50))  # Dynamic ef for better recall
            labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=actual_k)
            
            results = []
            for label, distance in zip(labels[0], distances[0]):
                if label in self.chunks:
                    similarity = 1 - distance  # Convert distance to similarity
                    results.append((self.chunks[label], similarity))
            
            return results
        except Exception as e:
            logging.warning(f"Error in HNSW search: {e}")
            return []
    
    def save(self, filepath: str):
        """Save the index and chunks (disabled to avoid pickle issues)"""
        # Temporarily disabled due to pickle serialization issues
        logging.info("HNSW vector store saving disabled (pickle issues)")
        pass
    
    def load(self, filepath: str):
        """Load the index and chunks"""
        try:
            self.index.load_index(f"{filepath}_index.bin")
            
            with open(f"{filepath}_chunks.pkl", 'rb') as f:
                serializable_chunks = pickle.load(f)
            
            # Reconstruct SemanticChunk objects
            self.chunks = {}
            for chunk_id, chunk_data in serializable_chunks.items():
                embedding = np.array(chunk_data['embedding']) if chunk_data['embedding'] is not None else None
                chunk = SemanticChunk(
                    content=chunk_data['content'],
                    metadata=chunk_data['metadata'],
                    embedding=embedding,
                    chunk_id=chunk_data['chunk_id'],
                    source=chunk_data['source'],
                    timestamp=chunk_data['timestamp'],
                    semantic_score=chunk_data['semantic_score']
                )
                self.chunks[chunk_id] = chunk
            
            self.next_id = max(self.chunks.keys()) + 1 if self.chunks else 0
        except Exception as e:
            logging.warning(f"Could not load HNSW vector store: {e}")
            self.chunks = {}
            self.next_id = 0

class QueryRewriter:
    """Advanced query rewriting for financial domain"""
    
    def __init__(self):
        self.financial_synonyms = {
            "profit": ["earnings", "income", "net income", "profit margin"],
            "revenue": ["sales", "turnover", "top line", "gross revenue"],
            "debt": ["liabilities", "borrowings", "leverage", "debt ratio"],
            "growth": ["expansion", "increase", "development", "appreciation"],
            "performance": ["returns", "results", "metrics", "KPIs"],
            "risk": ["volatility", "uncertainty", "exposure", "downside"],
            "valuation": ["price", "worth", "market cap", "enterprise value"]
        }
        
        self.query_templates = {
            "trend": [
                "What is the {metric} trend over time?",
                "How has {metric} changed recently?",
                "Show me {metric} historical performance"
            ],
            "comparison": [
                "How does {metric} compare to industry average?",
                "Compare {metric} with competitors",
                "{metric} vs peer analysis"
            ],
            "analysis": [
                "Analyze {metric} performance",
                "What factors affect {metric}?",
                "Explain {metric} implications"
            ]
        }
    
    def rewrite_query(self, query: str) -> QueryRewrite:
        """Rewrite query with financial domain knowledge"""
        query_lower = query.lower()
        rewritten_queries = [query]  # Include original
        
        # Identify query type and intent
        query_type = self._classify_query_type(query_lower)
        intent = self._extract_intent(query_lower)
        entities = self._extract_entities(query_lower)
        
        # Generate synonym-based rewrites
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    rewritten = query_lower.replace(term, synonym)
                    rewritten_queries.append(rewritten)
        
        # Generate template-based rewrites
        if intent in self.query_templates:
            for entity in entities:
                for template in self.query_templates[intent]:
                    rewritten = template.format(metric=entity)
                    rewritten_queries.append(rewritten)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_rewrites = []
        for q in rewritten_queries:
            if q not in seen:
                seen.add(q)
                unique_rewrites.append(q)
        
        return QueryRewrite(
            original_query=query,
            rewritten_queries=unique_rewrites,
            query_type=query_type,
            intent=intent,
            entities=entities
        )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query into financial categories"""
        if any(word in query for word in ["price", "chart", "technical", "moving average"]):
            return "technical"
        elif any(word in query for word in ["news", "announcement", "event"]):
            return "news"
        elif any(word in query for word in ["revenue", "profit", "earnings", "financial"]):
            return "financial"
        else:
            return "general"
    
    def _extract_intent(self, query: str) -> str:
        """Extract user intent from query"""
        if any(word in query for word in ["trend", "over time", "historical"]):
            return "trend"
        elif any(word in query for word in ["compare", "vs", "versus", "against"]):
            return "comparison"
        elif any(word in query for word in ["analyze", "explain", "why", "how"]):
            return "analysis"
        else:
            return "general"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract financial entities from query"""
        entities = []
        financial_terms = list(self.financial_synonyms.keys())
        
        for term in financial_terms:
            if term in query:
                entities.append(term)
        
        return entities

class HybridRetriever:
    """Hybrid search combining vector similarity and BM25"""
    
    def __init__(self, vector_store: HNSWVectorStore, embedding_model: SentenceTransformer):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25 = None
        self.chunks_text = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf_matrix = None
        
    def index_for_keyword_search(self, chunks: List[SemanticChunk]):
        """Index chunks for keyword-based search"""
        self.chunks_text = [chunk.content for chunk in chunks]
        
        # BM25 indexing
        tokenized_chunks = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # TF-IDF indexing
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks_text)
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> List[Tuple[SemanticChunk, float]]:
        """Perform hybrid search combining semantic and keyword search"""
        # Vector search
        query_embedding = self.embedding_model.encode([query])
        vector_results = self.vector_store.search(query_embedding[0], k=k*2)
        
        # Keyword search (BM25)
        bm25_scores = self.bm25.get_scores(query.lower().split()) if self.bm25 else []
        
        # TF-IDF search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Combine scores
        combined_results = {}
        
        # Add vector results
        for chunk, score in vector_results:
            chunk_id = int(chunk.chunk_id) if chunk.chunk_id else 0
            combined_results[chunk_id] = {
                'chunk': chunk,
                'vector_score': score,
                'bm25_score': 0,
                'tfidf_score': 0
            }
        
        # Add BM25 scores
        if len(bm25_scores) > 0:
            for i, score in enumerate(bm25_scores):
                # Convert score to scalar if it's an array
                score_val = float(score) if hasattr(score, '__len__') and len(score) == 1 else float(score)
                
                if i in combined_results:
                    combined_results[i]['bm25_score'] = score_val
                elif score_val > 0:  # Only add if there's a meaningful BM25 score
                    # Find corresponding chunk
                    chunk = self.vector_store.chunks.get(i)
                    if chunk:
                        combined_results[i] = {
                            'chunk': chunk,
                            'vector_score': 0,
                            'bm25_score': score_val,
                            'tfidf_score': 0
                        }
        
        # Add TF-IDF scores
        for i, score in enumerate(tfidf_similarities):
            if i in combined_results:
                combined_results[i]['tfidf_score'] = score
        
        # Calculate final hybrid scores
        final_results = []
        for result in combined_results.values():
            # Normalize scores
            vector_score = result['vector_score']
            
            # Safe BM25 normalization - handle numpy arrays properly
            if len(bm25_scores) > 0:
                max_bm25 = float(np.max(bm25_scores)) if hasattr(np.max(bm25_scores), '__len__') else float(max(bm25_scores))
                max_bm25 = max(max_bm25, 1.0)  # Avoid division by zero
                bm25_score = result['bm25_score'] / max_bm25
            else:
                bm25_score = 0
                
            tfidf_score = result['tfidf_score']
            
            # Weighted combination
            hybrid_score = float(alpha * vector_score + 
                           (1 - alpha) * 0.5 * (bm25_score + tfidf_score))
            
            final_results.append((result['chunk'], hybrid_score))
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

class Reranker:
    """Advanced reranking using cross-encoder models"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(model_name)
            self.available = True
        except ImportError:
            self.available = False
            logging.warning("CrossEncoder not available, using fallback reranking")
    
    def rerank(self, query: str, chunks_with_scores: List[Tuple[SemanticChunk, float]], 
              top_k: int = 5) -> List[Tuple[SemanticChunk, float]]:
        """Rerank chunks using cross-encoder"""
        if not self.available or len(chunks_with_scores) <= 1:
            return chunks_with_scores[:top_k]
        
        # Prepare query-chunk pairs
        pairs = [(query, chunk.content) for chunk, _ in chunks_with_scores]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Combine with original scores
        reranked_results = []
        for i, (chunk, original_score) in enumerate(chunks_with_scores):
            combined_score = 0.6 * cross_scores[i] + 0.4 * original_score
            reranked_results.append((chunk, combined_score))
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]

class AutocutGenerator:
    """Autocut technique for dynamic context selection"""
    
    def __init__(self, relevance_threshold: float = 0.1):
        self.relevance_threshold = relevance_threshold
    
    def autocut_context(self, chunks_with_scores: List[Tuple[SemanticChunk, float]], 
                       max_tokens: int = 4000) -> List[SemanticChunk]:
        """Automatically cut context based on relevance and token limits"""
        if not chunks_with_scores:
            return []
        
        # Sort by relevance score
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)
        
        # Apply relevance threshold
        relevant_chunks = [
            chunk for chunk, score in sorted_chunks 
            if score >= self.relevance_threshold
        ]
        
        # Apply token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk in relevant_chunks:
            chunk_tokens = len(chunk.content.split())
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        
        return selected_chunks

class AdvancedRAGSystem:
    """Main RAG system orchestrating all components"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.current_model_name = embedding_model
        self._initialize_model(embedding_model)
        
        # Storage for different document types
        self.indexed_documents = {
            'financial_statements': [],
            'news': [],
            'stock_data': [],
            'company_info': []
        }
    
    def _initialize_model(self, embedding_model: str):
        """Initialize or reinitialize the embedding model and components"""
        model_config = EMBEDDING_MODELS.get(embedding_model, {"trust_remote_code": False})
        trust_remote_code = model_config.get("trust_remote_code", False)
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=trust_remote_code)
        except Exception as e:
            logging.error(f"Failed to load embedding model {embedding_model}: {e}")
            # Try local cache paths
            import os
            cache_paths = [
                "/app/.cache/transformers",
                "/app/.cache/huggingface", 
                os.path.expanduser("~/.cache/huggingface"),
                "./models"
            ]
            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    try:
                        os.environ['TRANSFORMERS_CACHE'] = cache_path
                        os.environ['HF_HOME'] = cache_path
                        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=trust_remote_code)
                        break
                    except Exception:
                        continue
            else:
                raise RuntimeError(f"Cannot load embedding model: {e}")
        
        # Get model dimension
        model_dimension = EMBEDDING_MODELS.get(embedding_model, {}).get('dimension', 384)
        
        # Initialize components
        self.chunker = SemanticChunker(embedding_model)
        self.vector_store = HNSWVectorStore(dimension=model_dimension)
        self.query_rewriter = QueryRewriter()
        self.retriever = HybridRetriever(self.vector_store, self.embedding_model)
        self.reranker = Reranker()
        self.autocut = AutocutGenerator()
    
    def change_embedding_model(self, new_model: str):
        """Change the embedding model and reinitialize components"""
        if new_model != self.current_model_name:
            self.current_model_name = new_model
            self._initialize_model(new_model)
            
            # Re-index all existing documents with the new model
            self._reindex_all_documents()
            
            return True
        return False
    
    def _reindex_all_documents(self):
        """Re-index all stored documents with the new embedding model"""
        all_chunks = []
        
        # Collect all chunks from indexed documents
        for doc_type, chunks in self.indexed_documents.items():
            for chunk in chunks:
                # Re-embed with new model
                chunk.embedding = self.embedding_model.encode([chunk.content])[0]
                all_chunks.append(chunk)
        
        # Clear and rebuild vector store
        model_dimension = EMBEDDING_MODELS.get(self.current_model_name, {}).get('dimension', 384)
        self.vector_store.reinitialize(model_dimension)
        
        if all_chunks:
            self.vector_store.add_chunks(all_chunks)
            self.retriever.index_for_keyword_search(all_chunks)
    
    def index_financial_data(self, financial_data: Dict[str, Any], 
                           company_symbol: str, data_type: str):
        """Index financial data with appropriate chunking"""
        chunks = []
        timestamp = datetime.now()
        
        if data_type == 'financial_statements':
            chunks.extend(self._chunk_financial_statements(financial_data, company_symbol, timestamp))
        elif data_type == 'news':
            chunks.extend(self._chunk_news_data(financial_data, company_symbol, timestamp))
        elif data_type == 'stock_data':
            chunks.extend(self._chunk_stock_data(financial_data, company_symbol, timestamp))
        elif data_type == 'company_info':
            chunks.extend(self._chunk_company_info(financial_data, company_symbol, timestamp))
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        # Index for keyword search
        all_chunks = []
        for doc_type in self.indexed_documents.values():
            all_chunks.extend(doc_type)
        all_chunks.extend(chunks)
        
        self.retriever.index_for_keyword_search(all_chunks)
        
        # Store chunks
        self.indexed_documents[data_type].extend(chunks)
        
        return len(chunks)
    
    def query(self, question: str, k: int = 5, use_autocut: bool = True) -> Dict[str, Any]:
        """Main query interface with advanced RAG pipeline"""
        
        # Step 1: Query Rewriting
        query_rewrites = self.query_rewriter.rewrite_query(question)
        
        # Step 2: Hybrid Search for each rewritten query
        all_results = []
        for rewritten_query in query_rewrites.rewritten_queries[:3]:  # Limit to top 3 rewrites
            results = self.retriever.hybrid_search(rewritten_query, k=k*2)
            all_results.extend(results)
        
        # Deduplicate results
        seen_chunks = set()
        unique_results = []
        for chunk, score in all_results:
            chunk_content = chunk.content
            if chunk_content not in seen_chunks:
                seen_chunks.add(chunk_content)
                unique_results.append((chunk, score))
        
        # Step 3: Reranking
        reranked_results = self.reranker.rerank(question, unique_results, top_k=k*2)
        
        # Step 4: Autocut (if enabled)
        if use_autocut:
            final_chunks = self.autocut.autocut_context(reranked_results)
        else:
            final_chunks = [chunk for chunk, _ in reranked_results[:k]]
        
        # Prepare response
        context_text = "\n\n".join([chunk.content for chunk in final_chunks])
        
        return {
            'context': context_text,
            'chunks': final_chunks,
            'query_rewrites': query_rewrites,
            'num_chunks_retrieved': len(final_chunks),
            'retrieval_scores': [score for _, score in reranked_results[:len(final_chunks)]]
        }
    
    def _chunk_financial_statements(self, data: Dict[str, Any], symbol: str, timestamp: datetime) -> List[SemanticChunk]:
        """Chunk financial statements data"""
        chunks = []
        
        for statement_type, df in data.items():
            if df is not None and not df.empty:
                # Convert each row to text
                for metric in df.index:
                    values_text = []
                    for period in df.columns:
                        value = df.loc[metric, period]
                        if pd.notna(value):
                            # Handle both datetime and string periods
                            if hasattr(period, 'strftime'):
                                period_str = period.strftime('%Y-%m-%d')
                            else:
                                period_str = str(period)
                            values_text.append(f"{period_str}: ${value:,.0f}")
                    
                    if values_text:
                        content = f"{statement_type.replace('_', ' ').title()} - {metric}: " + "; ".join(values_text)
                        metadata = {
                            'symbol': symbol,
                            'statement_type': statement_type,
                            'metric': metric,
                            'data_type': 'financial_statements',
                            'timestamp': timestamp
                        }
                        
                        chunk_list = self.chunker.chunk_text(content, metadata)
                        chunks.extend(chunk_list)
        
        return chunks
    
    def _chunk_news_data(self, news_list: List[Dict], symbol: str, timestamp: datetime) -> List[SemanticChunk]:
        """Chunk news articles"""
        chunks = []
        
        for article in news_list:
            content_parts = []
            if article.get('title'):
                content_parts.append(f"Title: {article['title']}")
            if article.get('summary'):
                content_parts.append(f"Summary: {article['summary']}")
            if article.get('content'):
                content_parts.append(f"Content: {article['content']}")
            
            if content_parts:
                content = "\n".join(content_parts)
                metadata = {
                    'symbol': symbol,
                    'data_type': 'news',
                    'title': article.get('title', ''),
                    'publisher': article.get('publisher', ''),
                    'published': article.get('published', ''),
                    'timestamp': timestamp
                }
                
                chunk_list = self.chunker.chunk_text(content, metadata)
                chunks.extend(chunk_list)
        
        return chunks
    
    def _chunk_stock_data(self, stock_df: pd.DataFrame, symbol: str, timestamp: datetime) -> List[SemanticChunk]:
        """Chunk stock price data"""
        chunks = []
        
        # Create summaries for different time periods
        periods = {
            'recent_week': stock_df.tail(7),
            'recent_month': stock_df.tail(30),
            'recent_quarter': stock_df.tail(90)
        }
        
        for period_name, period_data in periods.items():
            if not period_data.empty:
                start_price = period_data['Close'].iloc[0]
                end_price = period_data['Close'].iloc[-1]
                high_price = period_data['High'].max()
                low_price = period_data['Low'].min()
                avg_volume = period_data['Volume'].mean()
                
                content = f"""Stock Performance for {symbol} ({period_name.replace('_', ' ').title()}):
                Starting Price: ${start_price:.2f}
                Ending Price: ${end_price:.2f}
                Highest Price: ${high_price:.2f}
                Lowest Price: ${low_price:.2f}
                Average Volume: {avg_volume:,.0f}
                Price Change: {((end_price - start_price) / start_price * 100):+.2f}%"""
                
                metadata = {
                    'symbol': symbol,
                    'data_type': 'stock_data',
                    'period': period_name,
                    'start_date': period_data.index[0].strftime('%Y-%m-%d'),
                    'end_date': period_data.index[-1].strftime('%Y-%m-%d'),
                    'timestamp': timestamp
                }
                
                chunk_list = self.chunker.chunk_text(content, metadata)
                chunks.extend(chunk_list)
        
        return chunks
    
    def _chunk_company_info(self, info_dict: Dict[str, Any], symbol: str, timestamp: datetime) -> List[SemanticChunk]:
        """Chunk company information"""
        chunks = []
        
        # Group related information
        info_groups = {
            'basic_info': ['longName', 'sector', 'industry', 'website', 'city', 'country'],
            'financial_metrics': ['marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE', 'priceToBook'],
            'business_summary': ['longBusinessSummary'],
            'key_stats': ['sharesOutstanding', 'floatShares', 'beta', 'dividendYield']
        }
        
        for group_name, fields in info_groups.items():
            content_parts = []
            for field in fields:
                if field in info_dict and info_dict[field] is not None:
                    value = info_dict[field]
                    if isinstance(value, (int, float)):
                        if field in ['marketCap', 'enterpriseValue']:
                            content_parts.append(f"{field}: ${value:,.0f}")
                        else:
                            content_parts.append(f"{field}: {value}")
                    else:
                        content_parts.append(f"{field}: {value}")
            
            if content_parts:
                content = f"Company Information for {symbol} ({group_name.replace('_', ' ').title()}):\n" + "\n".join(content_parts)
                metadata = {
                    'symbol': symbol,
                    'data_type': 'company_info',
                    'info_group': group_name,
                    'timestamp': timestamp
                }
                
                chunk_list = self.chunker.chunk_text(content, metadata)
                chunks.extend(chunk_list)
        
        return chunks
    
    def save_index(self, filepath: str):
        """Save the entire RAG system (disabled to avoid pickle issues)"""
        # Temporarily disabled due to pickle serialization issues
        logging.info("RAG index saving disabled (pickle issues)")
        pass
    
    def load_index(self, filepath: str):
        """Load the entire RAG system"""
        try:
            self.vector_store.load(filepath)
            
            # Load additional data
            with open(f"{filepath}_system.pkl", 'rb') as f:
                data = pickle.load(f)
                
                # Reconstruct SemanticChunk objects from serialized data
                self.indexed_documents = {}
                for doc_type, chunk_data_list in data['indexed_documents'].items():
                    self.indexed_documents[doc_type] = []
                    for chunk_data in chunk_data_list:
                        chunk = SemanticChunk(
                            content=chunk_data['content'],
                            metadata=chunk_data['metadata'],
                            chunk_id=chunk_data['chunk_id'],
                            source=chunk_data['source'],
                            timestamp=chunk_data['timestamp'],
                            semantic_score=chunk_data['semantic_score']
                        )
                        self.indexed_documents[doc_type].append(chunk)
                
                # Restore chunker config
                config = data['chunker_config']
                self.chunker.coherence_threshold = config['coherence_threshold']
                self.chunker.max_chunk_size = config['max_chunk_size'] 
                self.chunker.min_chunk_size = config['min_chunk_size']
            
            # Rebuild keyword search index
            all_chunks = []
            for doc_type in self.indexed_documents.values():
                all_chunks.extend(doc_type)
            
            if all_chunks:
                self.retriever.index_for_keyword_search(all_chunks)
                
        except Exception as e:
            logging.warning(f"Could not load RAG system: {e}")
            # Initialize empty if loading fails
            self.indexed_documents = {
                'financial_statements': [],
                'news': [],
                'stock_data': [],
                'company_info': []
            }

class MultiModelRAGSystem:
    """Advanced RAG system that maintains multiple embedding models simultaneously"""
    
    def __init__(self):
        self.models = {}
        self.active_model_key = "sentence-transformers/all-MiniLM-L6-v2"
        self.ensemble_weights = {}
        
        # Initialize all available models
        for model_key, model_info in EMBEDDING_MODELS.items():
            self._initialize_model(model_key)
            
    def _initialize_model(self, model_key: str):
        """Initialize a specific embedding model"""
        try:
            logging.info(f"Initializing model: {EMBEDDING_MODELS[model_key]['name']}")
            rag_system = AdvancedRAGSystem(model_key)
            
            self.models[model_key] = {
                'rag_system': rag_system,
                'model_info': EMBEDDING_MODELS[model_key],
                'is_indexed': False,
                'last_updated': None,
                'indexed_data_hash': {},  # Track what data has been indexed
                'performance_metrics': {
                    'total_queries': 0,
                    'avg_retrieval_time': 0.0,
                    'avg_relevance_score': 0.0
                }
            }
            
            # Set default ensemble weight
            self.ensemble_weights[model_key] = 1.0
            
            logging.info(f"✅ Model {EMBEDDING_MODELS[model_key]['name']} initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize model {model_key}: {e}")
            self.models[model_key] = {
                'rag_system': None,
                'model_info': EMBEDDING_MODELS[model_key],
                'is_indexed': False,
                'error': str(e)
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of successfully initialized model keys"""
        return [key for key, info in self.models.items() 
                if info.get('rag_system') is not None]
    
    def set_active_model(self, model_key: str):
        """Set the active model for single-model operations"""
        if model_key in self.get_available_models():
            self.active_model_key = model_key
            return True
        return False
    
    def get_active_model(self) -> str:
        """Get the current active model key"""
        return self.active_model_key
    
    def index_financial_data(self, financial_data: Dict[str, Any], 
                           company_symbol: str, data_type: str) -> Dict[str, int]:
        """Index financial data in all available models simultaneously"""
        results = {}
        data_hash = hash(str(financial_data))
        
        for model_key in self.get_available_models():
            model_info = self.models[model_key]
            
            # Check if this data has already been indexed for this model
            if model_info['indexed_data_hash'].get(f"{company_symbol}_{data_type}") == data_hash:
                results[model_key] = 0  # Already indexed
                continue
                
            try:
                rag_system = model_info['rag_system']
                chunks_count = rag_system.index_financial_data(financial_data, company_symbol, data_type)
                
                # Update tracking info
                model_info['is_indexed'] = True
                model_info['last_updated'] = datetime.now()
                model_info['indexed_data_hash'][f"{company_symbol}_{data_type}"] = data_hash
                
                results[model_key] = chunks_count
                
                logging.info(f"Indexed {chunks_count} chunks in {model_info['model_info']['name']}")
                
            except Exception as e:
                logging.error(f"Error indexing in model {model_key}: {e}")
                results[model_key] = -1
                
        return results
    
    def query_single_model(self, question: str, model_key: str = None, k: int = 5) -> Dict[str, Any]:
        """Query a specific model"""
        if model_key is None:
            model_key = self.active_model_key
            
        if model_key not in self.get_available_models():
            return {"error": f"Model {model_key} not available"}
            
        model_info = self.models[model_key]
        rag_system = model_info['rag_system']
        
        import time
        start_time = time.time()
        
        try:
            result = rag_system.query(question, k=k, use_autocut=True)
            
            # Update performance metrics
            retrieval_time = time.time() - start_time
            model_info['performance_metrics']['total_queries'] += 1
            
            # Update running average of retrieval time
            prev_avg = model_info['performance_metrics']['avg_retrieval_time']
            total_queries = model_info['performance_metrics']['total_queries']
            model_info['performance_metrics']['avg_retrieval_time'] = (
                (prev_avg * (total_queries - 1) + retrieval_time) / total_queries
            )
            
            # Add metadata
            result['model_used'] = model_key
            result['model_name'] = model_info['model_info']['name']
            result['retrieval_time'] = retrieval_time
            result['rag_enhanced'] = True
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_key,
                "rag_enhanced": False
            }
    
    def query_ensemble(self, question: str, k: int = 5, 
                      combination_method: str = "weighted_score") -> Dict[str, Any]:
        """Query all models and combine results using ensemble method"""
        available_models = self.get_available_models()
        
        if len(available_models) < 2:
            # Fall back to single model if only one available
            return self.query_single_model(question, k=k)
        
        model_results = {}
        all_chunks = []
        
        # Query each model
        for model_key in available_models:
            result = self.query_single_model(question, model_key, k=k*2)  # Get more for ensemble
            if not result.get('error'):
                model_results[model_key] = result
                
                # Collect chunks with model attribution
                for i, chunk in enumerate(result.get('chunks', [])):
                    score = result.get('retrieval_scores', [0])[i] if i < len(result.get('retrieval_scores', [])) else 0
                    weight = self.ensemble_weights.get(model_key, 1.0)
                    
                    all_chunks.append({
                        'chunk': chunk,
                        'model_key': model_key,
                        'model_name': self.models[model_key]['model_info']['name'],
                        'original_score': score,
                        'weighted_score': score * weight,
                        'model_weight': weight
                    })
        
        if not all_chunks:
            return {"error": "No results from any model", "rag_enhanced": False}
        
        # Combine results based on method
        if combination_method == "weighted_score":
            # Sort by weighted score
            all_chunks.sort(key=lambda x: x['weighted_score'], reverse=True)
        elif combination_method == "round_robin":
            # Interleave results from different models
            all_chunks = self._round_robin_combine(all_chunks, available_models)
        
        # Select top k unique chunks
        final_chunks = []
        seen_content = set()
        
        for chunk_info in all_chunks[:k*3]:  # Check more for uniqueness
            content = chunk_info['chunk'].content
            if content not in seen_content and len(final_chunks) < k:
                seen_content.add(content)
                final_chunks.append(chunk_info)
        
        # Create combined context
        context_parts = []
        retrieval_scores = []
        model_contributions = {}
        
        for chunk_info in final_chunks:
            context_parts.append(chunk_info['chunk'].content)
            retrieval_scores.append(chunk_info['weighted_score'])
            
            model_name = chunk_info['model_name']
            if model_name not in model_contributions:
                model_contributions[model_name] = 0
            model_contributions[model_name] += 1
        
        return {
            'context': '\n\n'.join(context_parts),
            'chunks': [info['chunk'] for info in final_chunks],
            'num_chunks_retrieved': len(final_chunks),
            'retrieval_scores': retrieval_scores,
            'model_contributions': model_contributions,
            'ensemble_method': combination_method,
            'models_used': list(model_results.keys()),
            'rag_enhanced': True,
            'is_ensemble': True
        }
    
    def _round_robin_combine(self, all_chunks: List[Dict], available_models: List[str]) -> List[Dict]:
        """Combine chunks using round-robin from each model"""
        model_chunks = {model: [] for model in available_models}
        
        # Group chunks by model
        for chunk_info in all_chunks:
            model_chunks[chunk_info['model_key']].append(chunk_info)
        
        # Round-robin selection
        combined = []
        max_per_model = max(len(chunks) for chunks in model_chunks.values())
        
        for i in range(max_per_model):
            for model_key in available_models:
                if i < len(model_chunks[model_key]):
                    combined.append(model_chunks[model_key][i])
        
        return combined
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models"""
        stats = {
            'active_model': self.active_model_key,
            'available_models': self.get_available_models(),
            'model_details': {}
        }
        
        for model_key, model_info in self.models.items():
            if model_info.get('rag_system'):
                rag_system = model_info['rag_system']
                stats['model_details'][model_key] = {
                    'name': model_info['model_info']['name'],
                    'dimension': model_info['model_info']['dimension'],
                    'total_chunks': len(rag_system.vector_store.chunks),
                    'is_indexed': model_info['is_indexed'],
                    'last_updated': model_info['last_updated'],
                    'performance_metrics': model_info['performance_metrics'],
                    'indexed_documents': {
                        doc_type: len(chunks) 
                        for doc_type, chunks in rag_system.indexed_documents.items()
                    }
                }
            else:
                stats['model_details'][model_key] = {
                    'name': model_info['model_info']['name'],
                    'error': model_info.get('error', 'Unknown error'),
                    'available': False
                }
        
        return stats
    
    def set_ensemble_weights(self, weights: Dict[str, float]):
        """Set ensemble weights for different models"""
        for model_key, weight in weights.items():
            if model_key in self.models:
                self.ensemble_weights[model_key] = weight
    
    def clear_model_cache(self, model_key: str = None):
        """Clear cache for specific model or all models"""
        if model_key:
            if model_key in self.models and self.models[model_key].get('rag_system'):
                rag_system = self.models[model_key]['rag_system']
                rag_system.vector_store = HNSWVectorStore(
                    dimension=EMBEDDING_MODELS[model_key]['dimension']
                )
                rag_system.indexed_documents = {
                    'financial_statements': [],
                    'news': [],
                    'stock_data': [],
                    'company_info': []
                }
                self.models[model_key]['is_indexed'] = False
                self.models[model_key]['indexed_data_hash'] = {}
        else:
            # Clear all models
            for model_key in self.get_available_models():
                self.clear_model_cache(model_key)

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = AdvancedRAGSystem()
    
    # Example data indexing
    sample_financial_data = {
        'quarterly_financials': pd.DataFrame({
            '2024-01-01': {'Total Revenue': 100000000, 'Net Income': 20000000},
            '2023-10-01': {'Total Revenue': 95000000, 'Net Income': 18000000}
        }).T
    }
    
    sample_news = [
        {
            'title': 'Company Reports Strong Q4 Earnings',
            'summary': 'The company exceeded analyst expectations with revenue growth of 15%',
            'content': 'Full article content here...'
        }
    ]
    
    # Index the data
    rag.index_financial_data(sample_financial_data, 'AAPL', 'financial_statements')
    rag.index_financial_data(sample_news, 'AAPL', 'news')
    
    # Query the system
    result = rag.query("What was the revenue growth last quarter?")
    print("Context:", result['context'])
    print("Query Rewrites:", result['query_rewrites'].rewritten_queries)