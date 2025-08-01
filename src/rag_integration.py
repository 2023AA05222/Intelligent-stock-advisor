"""
RAG Integration Module for Financial Analysis Streamlit App
Integrates the advanced RAG system with the existing application
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from datetime import datetime
import json

try:
    from .rag_system import AdvancedRAGSystem, SemanticChunk
    RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAG system not available: {e}")
    RAG_AVAILABLE = False

class RAGIntegration:
    """Integration layer between RAG system and Streamlit app"""
    
    def __init__(self, cache_dir: str = "rag_cache"):
        self.cache_dir = cache_dir
        self.rag_system = None
        self.is_initialized = False
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        if RAG_AVAILABLE:
            self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize the RAG system"""
        try:
            self.rag_system = AdvancedRAGSystem()
            
            # Try to load existing index (disabled for now due to pickle issues)
            # index_path = os.path.join(self.cache_dir, "financial_rag")
            # if os.path.exists(f"{index_path}_index.bin"):
            #     try:
            #         self.rag_system.load_index(index_path)
            #         st.success("🔍 RAG system loaded from cache")
            #     except Exception as e:
            #         st.warning(f"Could not load RAG cache: {e}")
            #         self.rag_system = AdvancedRAGSystem()  # Reinitialize
            
            self.is_initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if RAG system is available"""
        return RAG_AVAILABLE and self.is_initialized
    
    def index_stock_data(self, symbol: str, stock_data: pd.DataFrame, 
                        stock_info: Dict[str, Any], news_data: List[Dict], 
                        financial_statements: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, int]:
        """Index all available data for a stock symbol"""
        if not self.is_available():
            return {"error": "RAG system not available"}
        
        indexed_counts = {}
        
        try:
            # Index stock price data
            if stock_data is not None and not stock_data.empty:
                count = self.rag_system.index_financial_data(stock_data, symbol, 'stock_data')
                indexed_counts['stock_data'] = count
            
            # Index company information
            if stock_info:
                count = self.rag_system.index_financial_data(stock_info, symbol, 'company_info')
                indexed_counts['company_info'] = count
            
            # Index news data
            if news_data:
                count = self.rag_system.index_financial_data(news_data, symbol, 'news')
                indexed_counts['news'] = count
            
            # Index financial statements
            if financial_statements:
                count = self.rag_system.index_financial_data(financial_statements, symbol, 'financial_statements')
                indexed_counts['financial_statements'] = count
            
            # Save the updated index (disabled for now due to pickle issues)
            # self._save_index()
            
            return indexed_counts
            
        except Exception as e:
            st.error(f"Error indexing data: {e}")
            return {"error": str(e)}
    
    def query_rag(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.is_available():
            return {
                "error": "RAG system not available",
                "context": "",
                "rag_enhanced": False
            }
        
        try:
            # Check if we have any indexed data
            total_chunks = len(self.rag_system.vector_store.chunks)
            if total_chunks == 0:
                return {
                    "error": "No data indexed yet",
                    "context": "",
                    "rag_enhanced": False
                }
            
            # Limit k to available chunks
            actual_k = min(k, total_chunks)
            result = self.rag_system.query(question, k=actual_k, use_autocut=True)
            
            # Add metadata for the UI
            result['rag_enhanced'] = True
            result['query_timestamp'] = datetime.now()
            
            return result
            
        except Exception as e:
            logging.warning(f"Error querying RAG system: {e}")
            return {
                "error": str(e),
                "context": "",
                "rag_enhanced": False
            }
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data"""
        if not self.is_available():
            return {"error": "RAG system not available"}
        
        try:
            stats = {
                "total_chunks": len(self.rag_system.vector_store.chunks),
                "indexed_documents": {}
            }
            
            for doc_type, chunks in self.rag_system.indexed_documents.items():
                stats["indexed_documents"][doc_type] = len(chunks)
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def _save_index(self):
        """Save the RAG index to cache (disabled)"""
        # Temporarily disabled due to pickle serialization issues
        pass
    
    def clear_cache(self):
        """Clear the RAG cache"""
        try:
            index_path = os.path.join(self.cache_dir, "financial_rag")
            
            # Remove cache files
            for ext in ["_index.bin", "_chunks.pkl", "_system.pkl"]:
                file_path = f"{index_path}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Reinitialize RAG system
            if RAG_AVAILABLE:
                self.rag_system = AdvancedRAGSystem()
                self.is_initialized = True
            
            st.success("RAG cache cleared successfully")
            
        except Exception as e:
            st.error(f"Error clearing RAG cache: {e}")

def create_basic_financial_context(symbol: str, 
                                 stock_data: pd.DataFrame, 
                                 stock_info: Dict[str, Any], 
                                 news_data: List[Dict], 
                                 financial_statements: Optional[Dict[str, pd.DataFrame]]) -> str:
    """Create basic financial context without importing streamlit_app"""
    context_parts = []
    
    # Basic stock information
    if stock_info:
        context_parts.append(f"=== STOCK INFORMATION FOR {symbol} ===")
        for key, value in stock_info.items():
            if value is not None:
                context_parts.append(f"{key}: {value}")
        context_parts.append("")
    
    # Stock price data - include ALL data for comprehensive analysis
    if stock_data is not None and not stock_data.empty:
        context_parts.append("=== COMPLETE PRICE DATA ===")
        context_parts.append(f"Total data points: {len(stock_data)}")
        context_parts.append(f"Date range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
        context_parts.append("")
        
        # Include all price data for comprehensive analysis
        context_parts.append("Full price history:")
        for idx, row in stock_data.iterrows():
            context_parts.append(f"{idx.strftime('%Y-%m-%d %H:%M:%S')}: Open={row['Open']:.2f}, High={row['High']:.2f}, Low={row['Low']:.2f}, Close={row['Close']:.2f}, Volume={row['Volume']:,}")
        context_parts.append("")
    
    # Financial statements
    if financial_statements:
        context_parts.append("=== FINANCIAL STATEMENTS ===")
        for statement_type, df in financial_statements.items():
            if df is not None and not df.empty:
                context_parts.append(f"\n--- {statement_type.replace('_', ' ').title()} ---")
                # Show latest period data
                if len(df.columns) > 0:
                    latest_period = df.columns[0]
                    context_parts.append(f"Period: {latest_period.strftime('%Y-%m-%d')}")
                    for metric in df.index[:10]:  # Limit to top 10 metrics
                        value = df.loc[metric, latest_period]
                        if pd.notna(value):
                            context_parts.append(f"{metric}: {value:,.0f}")
        context_parts.append("")
    
    # News articles
    if news_data:
        context_parts.append("=== NEWS ARTICLES ===")
        for i, article in enumerate(news_data[:5], 1):  # Limit to 5 articles
            context_parts.append(f"\n{i}. {article.get('title', 'No title')}")
            if article.get('summary'):
                context_parts.append(f"   Summary: {article['summary']}")
        context_parts.append("")
    
    return "\n".join(context_parts)

def enhanced_financial_context_with_rag(rag_integration: RAGIntegration, 
                                       symbol: str, 
                                       stock_data: pd.DataFrame, 
                                       stock_info: Dict[str, Any], 
                                       news_data: List[Dict], 
                                       financial_statements: Optional[Dict[str, pd.DataFrame]], 
                                       user_question: str) -> str:
    """Create enhanced financial context using RAG"""
    
    # First, index all the data
    if rag_integration.is_available():
        indexed_counts = rag_integration.index_stock_data(
            symbol, stock_data, stock_info, news_data, financial_statements
        )
        
        # Query RAG system for relevant context
        rag_result = rag_integration.query_rag(user_question, k=8)
        
        if rag_result.get('rag_enhanced'):
            # Use RAG-enhanced context
            context_parts = [
                f"=== RAG-ENHANCED CONTEXT FOR {symbol} ===",
                f"Query: {user_question}",
                "",
                "=== MOST RELEVANT INFORMATION ===",
                rag_result['context'],
                "",
                f"=== RAG METADATA ===",
                f"Chunks retrieved: {rag_result['num_chunks_retrieved']}",
                f"Query rewrites used: {len(rag_result['query_rewrites'].rewritten_queries)}",
                f"Retrieval scores: {[f'{score:.3f}' for score in rag_result['retrieval_scores'][:3]]}",
                ""
            ]
            
            return "\n".join(context_parts)
    
    # Fallback to basic context creation
    return create_basic_financial_context(symbol, stock_data, stock_info, news_data, financial_statements)

@st.cache_resource
def get_rag_integration() -> RAGIntegration:
    """Get or create RAG integration instance"""
    return RAGIntegration()

def display_rag_status_sidebar():
    """Display RAG system status in sidebar"""
    rag_integration = get_rag_integration()
    
    with st.sidebar:
        st.markdown("### 🔍 RAG System Status")
        
        if rag_integration.is_available():
            st.success("✅ RAG System Active")
            
            # Show stats
            stats = rag_integration.get_rag_stats()
            if "error" not in stats:
                st.metric("Total Chunks", stats["total_chunks"])
                
                with st.expander("📊 Indexed Documents"):
                    for doc_type, count in stats["indexed_documents"].items():
                        st.metric(doc_type.replace('_', ' ').title(), count)
            
            # Clear cache button
            if st.button("🗑️ Clear RAG Cache"):
                rag_integration.clear_cache()
                st.rerun()
                
        else:
            st.warning("⚠️ RAG System Unavailable")
            st.info("Install dependencies: `pip install -r requirements-rag.txt`")

def rag_enhanced_prompt_modification(base_prompt: str, rag_context: str, user_question: str) -> str:
    """Modify the base prompt to work with RAG-enhanced context"""
    
    if "RAG-ENHANCED CONTEXT" in rag_context:
        # This is RAG-enhanced context
        enhanced_prompt = f"""{base_prompt}

SPECIAL INSTRUCTIONS FOR RAG-ENHANCED ANALYSIS:
The context below has been enhanced using Retrieval-Augmented Generation (RAG) techniques including:
- Semantic chunking for coherent information segments
- HNSW vector indexing for efficient similarity search  
- Hybrid search combining semantic and keyword matching
- Query rewriting for comprehensive information retrieval
- Cross-encoder reranking for relevance optimization
- Autocut for optimal context selection

This means the most relevant information for the user's question has been automatically identified and prioritized. Pay special attention to the "MOST RELEVANT INFORMATION" section as it contains the most pertinent data for answering the user's specific question.

ENHANCED FINANCIAL DATA CONTEXT:
{rag_context}

USER QUESTION: {user_question}

Please provide your analysis giving priority to the RAG-retrieved relevant information while still following the mandatory analysis framework."""

        return enhanced_prompt.format(rag_context=rag_context, user_question=user_question)
    
    else:
        # Regular context - format with both parameters
        return base_prompt.format(context=rag_context, user_question=user_question)

# Integration utilities for the main app
def auto_index_on_data_fetch(symbol: str, stock_data: pd.DataFrame, 
                            stock_info: Dict[str, Any], news_data: List[Dict],
                            financial_statements: Optional[Dict[str, pd.DataFrame]] = None):
    """Automatically index data when it's fetched"""
    rag_integration = get_rag_integration()
    
    if rag_integration.is_available():
        with st.spinner("🔍 Indexing data for RAG..."):
            indexed_counts = rag_integration.index_stock_data(
                symbol, stock_data, stock_info, news_data, financial_statements
            )
            
            if "error" not in indexed_counts:
                total_chunks = sum(indexed_counts.values())
                if total_chunks > 0:
                    st.success(f"📚 Indexed {total_chunks} chunks for enhanced AI analysis")

def get_rag_enhanced_context(symbol: str, stock_data: pd.DataFrame, 
                           stock_info: Dict[str, Any], news_data: List[Dict],
                           financial_statements: Optional[Dict[str, pd.DataFrame]], 
                           user_question: str) -> Tuple[str, bool]:
    """Get either RAG-enhanced or regular context"""
    rag_integration = get_rag_integration()
    
    if rag_integration.is_available():
        context = enhanced_financial_context_with_rag(
            rag_integration, symbol, stock_data, stock_info, 
            news_data, financial_statements, user_question
        )
        return context, True
    else:
        # Import and use the original function
        context = create_basic_financial_context(symbol, stock_data, stock_info, news_data, financial_statements)
        return context, False