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
    from .rag_system import AdvancedRAGSystem, SemanticChunk, EMBEDDING_MODELS, MultiModelRAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAG system not available: {e}")
    RAG_AVAILABLE = False
    EMBEDDING_MODELS = {}

class RAGIntegration:
    """Integration layer between MultiModel RAG system and Streamlit app"""
    
    def __init__(self, cache_dir: str = "rag_cache"):
        self.cache_dir = cache_dir
        self.multi_rag_system = None
        self.is_initialized = False
        self.query_mode = "single"  # "single", "ensemble", or "compare"
        self.ensemble_method = "weighted_score"  # "weighted_score" or "round_robin"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        if RAG_AVAILABLE:
            self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize the MultiModel RAG system"""
        try:
            self.multi_rag_system = MultiModelRAGSystem()
            self.is_initialized = True
            
            available_models = self.multi_rag_system.get_available_models()
            if available_models:
                st.success(f"🔍 Multi-model RAG system initialized with {len(available_models)} models")
                for model_key in available_models:
                    model_name = EMBEDDING_MODELS[model_key]['name']
                    st.info(f"✅ {model_name} ready")
            else:
                st.warning("⚠️ No embedding models available")
                
        except Exception as e:
            st.error(f"Failed to initialize multi-model RAG system: {e}")
            self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if RAG system is available"""
        return RAG_AVAILABLE and self.is_initialized
    
    def set_active_model(self, model_key: str) -> bool:
        """Set the active model for single-model operations"""
        if not self.is_available():
            return False
        
        success = self.multi_rag_system.set_active_model(model_key)
        if success:
            model_name = EMBEDDING_MODELS.get(model_key, {}).get('name', model_key)
            st.success(f"🎯 Active model set to {model_name}")
        return success
    
    def get_active_model(self) -> str:
        """Get the current active model"""
        if self.is_available():
            return self.multi_rag_system.get_active_model()
        return ""
    
    def get_available_models(self) -> List[str]:
        """Get available model keys"""
        if self.is_available():
            return self.multi_rag_system.get_available_models()
        return []
    
    def set_query_mode(self, mode: str):
        """Set query mode: 'single', 'ensemble', or 'compare'"""
        if mode in ['single', 'ensemble', 'compare']:
            self.query_mode = mode
    
    def get_query_mode(self) -> str:
        """Get current query mode"""
        return self.query_mode
    
    def set_ensemble_method(self, method: str):
        """Set ensemble combination method"""
        if method in ['weighted_score', 'round_robin']:
            self.ensemble_method = method
    
    def set_ensemble_weights(self, weights: Dict[str, float]):
        """Set ensemble model weights"""
        if self.is_available():
            self.multi_rag_system.set_ensemble_weights(weights)
    
    def index_stock_data(self, symbol: str, stock_data: pd.DataFrame, 
                        stock_info: Dict[str, Any], news_data: List[Dict], 
                        financial_statements: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """Index all available data for a stock symbol in all models"""
        if not self.is_available():
            return {"error": "Multi-model RAG system not available"}
        
        try:
            all_results = {}
            
            # Index stock price data
            if stock_data is not None and not stock_data.empty:
                results = self.multi_rag_system.index_financial_data(stock_data, symbol, 'stock_data')
                all_results['stock_data'] = results
            
            # Index company information
            if stock_info:
                results = self.multi_rag_system.index_financial_data(stock_info, symbol, 'company_info')
                all_results['company_info'] = results
            
            # Index news data
            if news_data:
                results = self.multi_rag_system.index_financial_data(news_data, symbol, 'news')
                all_results['news'] = results
            
            # Index financial statements
            if financial_statements:
                results = self.multi_rag_system.index_financial_data(financial_statements, symbol, 'financial_statements')
                all_results['financial_statements'] = results
            
            return all_results
            
        except Exception as e:
            st.error(f"Error indexing data across models: {e}")
            return {"error": str(e)}
    
    def query_rag(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system using the selected mode"""
        if not self.is_available():
            return {
                "error": "Multi-model RAG system not available",
                "context": "",
                "rag_enhanced": False
            }
        
        try:
            if self.query_mode == "single":
                result = self.multi_rag_system.query_single_model(question, k=k)
            elif self.query_mode == "ensemble":
                result = self.multi_rag_system.query_ensemble(
                    question, k=k, combination_method=self.ensemble_method
                )
            elif self.query_mode == "compare":
                # Return results from all models for comparison
                available_models = self.multi_rag_system.get_available_models()
                comparison_results = {}
                for model_key in available_models:
                    model_result = self.multi_rag_system.query_single_model(question, model_key, k=k)
                    comparison_results[model_key] = model_result
                
                # Create combined response for UI
                result = {
                    "comparison_results": comparison_results,
                    "rag_enhanced": True,
                    "is_comparison": True,
                    "models_compared": len(available_models)
                }
            else:
                return {"error": f"Invalid query mode: {self.query_mode}", "rag_enhanced": False}
            
            # Add metadata
            result['query_timestamp'] = datetime.now()
            result['query_mode'] = self.query_mode
            
            return result
            
        except Exception as e:
            logging.warning(f"Error querying multi-model RAG system: {e}")
            return {
                "error": str(e),
                "context": "",
                "rag_enhanced": False
            }
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get statistics about all models"""
        if not self.is_available():
            return {"error": "Multi-model RAG system not available"}
        
        try:
            return self.multi_rag_system.get_model_stats()
        except Exception as e:
            return {"error": str(e)}
    
    def _save_index(self):
        """Save the RAG index to cache (disabled)"""
        # Temporarily disabled due to pickle serialization issues
        pass
    
    def clear_cache(self, model_key: str = None):
        """Clear cache for specific model or all models"""
        try:
            if self.is_available():
                self.multi_rag_system.clear_model_cache(model_key)
                
                if model_key:
                    model_name = EMBEDDING_MODELS.get(model_key, {}).get('name', model_key)
                    st.success(f"Cache cleared for {model_name}")
                else:
                    st.success("All model caches cleared successfully")
            else:
                st.warning("Multi-model RAG system not available")
            
        except Exception as e:
            st.error(f"Error clearing cache: {e}")

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
    """Display multi-model RAG system status in sidebar"""
    rag_integration = get_rag_integration()
    
    with st.sidebar:
        st.markdown("### 🔍 Multi-Model RAG System")
        
        if rag_integration.is_available():
            st.success("✅ Multi-Model RAG Active")
            
            # Get stats for all models
            stats = rag_integration.get_rag_stats()
            if "error" not in stats:
                available_models = stats.get('available_models', [])
                active_model = stats.get('active_model', '')
                
                # Query Mode Selection
                st.markdown("#### 🎯 Query Mode")
                query_mode = st.selectbox(
                    "Select query mode:",
                    options=["single", "ensemble", "compare"],
                    index=["single", "ensemble", "compare"].index(rag_integration.get_query_mode()),
                    format_func=lambda x: {
                        "single": "🎯 Single Model",
                        "ensemble": "🤝 Ensemble (Combined)",
                        "compare": "⚖️ Compare Models"
                    }[x],
                    key="query_mode_select"
                )
                rag_integration.set_query_mode(query_mode)
                
                # Active Model Selection (for single mode)
                if query_mode == "single":
                    st.markdown("#### 🧠 Active Model")
                    model_options = {}
                    model_names = []
                    
                    for model_key in available_models:
                        if model_key in EMBEDDING_MODELS:
                            display_name = EMBEDDING_MODELS[model_key]['name']
                            model_options[display_name] = model_key
                            model_names.append(display_name)
                    
                    # Find current active model display name
                    current_display_name = None
                    for display_name, model_key in model_options.items():
                        if model_key == active_model:
                            current_display_name = display_name
                            break
                    
                    if current_display_name and current_display_name in model_names:
                        current_index = model_names.index(current_display_name)
                    else:
                        current_index = 0
                    
                    if model_names:
                        selected_display_name = st.selectbox(
                            "Choose active model:",
                            options=model_names,
                            index=current_index,
                            key="active_model_select"
                        )
                        
                        selected_model = model_options[selected_display_name]
                        
                        # Show model details
                        if selected_model in EMBEDDING_MODELS:
                            model_info = EMBEDDING_MODELS[selected_model]
                            st.caption(f"**Dimension:** {model_info['dimension']}")
                            st.caption(f"**Description:** {model_info['description']}")
                        
                        # Change active model if different
                        if selected_model != active_model:
                            rag_integration.set_active_model(selected_model)
                            st.rerun()
                
                # Ensemble Settings
                elif query_mode == "ensemble":
                    st.markdown("#### 🤝 Ensemble Settings")
                    
                    # Ensemble method
                    ensemble_method = st.selectbox(
                        "Combination method:",
                        options=["weighted_score", "round_robin"],
                        format_func=lambda x: {
                            "weighted_score": "📊 Weighted Score",
                            "round_robin": "🔄 Round Robin"
                        }[x],
                        key="ensemble_method_select"
                    )
                    rag_integration.set_ensemble_method(ensemble_method)
                    
                    # Model weights (if weighted_score)
                    if ensemble_method == "weighted_score":
                        st.markdown("##### Model Weights:")
                        weights = {}
                        for model_key in available_models:
                            if model_key in EMBEDDING_MODELS:
                                model_name = EMBEDDING_MODELS[model_key]['name']
                                weight = st.slider(
                                    f"{model_name}:",
                                    min_value=0.0,
                                    max_value=2.0,
                                    value=1.0,
                                    step=0.1,
                                    key=f"weight_{model_key}"
                                )
                                weights[model_key] = weight
                        
                        if st.button("Update Weights", key="update_weights"):
                            rag_integration.set_ensemble_weights(weights)
                            st.success("Weights updated!")
                
                # Model Statistics
                st.markdown("#### 📊 Model Statistics")
                
                # Summary metrics
                total_chunks_all_models = 0
                for model_key, model_details in stats.get('model_details', {}).items():
                    if model_details.get('available', True):
                        total_chunks_all_models += model_details.get('total_chunks', 0)
                
                col1, col2 = st.columns(2)
                col1.metric("Models", len(available_models))
                col2.metric("Total Chunks", total_chunks_all_models)
                
                # Detailed model stats
                with st.expander("📈 Detailed Model Stats"):
                    for model_key, model_details in stats.get('model_details', {}).items():
                        if model_details.get('available', True):
                            st.markdown(f"**{model_details['name']}**")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Chunks", model_details.get('total_chunks', 0))
                            col2.metric("Queries", model_details.get('performance_metrics', {}).get('total_queries', 0))
                            
                            avg_time = model_details.get('performance_metrics', {}).get('avg_retrieval_time', 0)
                            col3.metric("Avg Time", f"{avg_time:.3f}s" if avg_time else "0s")
                            
                            # Document breakdown
                            indexed_docs = model_details.get('indexed_documents', {})
                            if any(count > 0 for count in indexed_docs.values()):
                                for doc_type, count in indexed_docs.items():
                                    if count > 0:
                                        st.caption(f"  • {doc_type.replace('_', ' ').title()}: {count}")
                            
                            st.markdown("---")
                
                # Cache Management
                st.markdown("#### 🗑️ Cache Management")
                
                # Clear specific model cache
                if len(available_models) > 1:
                    clear_model = st.selectbox(
                        "Clear cache for:",
                        options=["All Models"] + [EMBEDDING_MODELS[key]['name'] for key in available_models],
                        key="clear_model_select"
                    )
                    
                    if st.button("🗑️ Clear Selected Cache"):
                        if clear_model == "All Models":
                            rag_integration.clear_cache()
                        else:
                            # Find model key by name
                            for model_key in available_models:
                                if EMBEDDING_MODELS[model_key]['name'] == clear_model:
                                    rag_integration.clear_cache(model_key)
                                    break
                        st.rerun()
                else:
                    if st.button("🗑️ Clear All Cache"):
                        rag_integration.clear_cache()
                        st.rerun()
                        
        else:
            st.warning("⚠️ Multi-Model RAG Unavailable")
            st.info("Install dependencies: `pip install -r requirements.txt`")

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
                # Calculate total chunks across all models and data types
                total_chunks = 0
                for data_type, model_results in indexed_counts.items():
                    if isinstance(model_results, dict):
                        for model_key, count in model_results.items():
                            if isinstance(count, int) and count > 0:
                                total_chunks += count
                
                if total_chunks > 0:
                    st.success(f"📚 Indexed {total_chunks} chunks across all models for enhanced AI analysis")

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