"""
Streamlit UI Orchestrator for Receipt Intelligence System.
"""

import streamlit as st
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add src to path for robust imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Modular UI Components
from components.dashboard import render_full_dashboard
from components.visuals import render_response_feed_item

# Core Business Logic
try:
    from ..utils.logging_config import logger, setup_logging
    from ..models import Receipt, ReceiptChunk
    from ..parsers import ReceiptParser
    from ..chunking import ReceiptChunker
    from ..vectorstore import VectorManager
    from ..query import QueryEngine
except (ImportError, ValueError):
    from utils.logging_config import logger, setup_logging
    from models import Receipt, ReceiptChunk
    from parsers import ReceiptParser
    from chunking import ReceiptChunker
    from vectorstore import VectorManager
    from query import QueryEngine

HISTORY_FILE = "data/query_history.json"

@st.cache_resource
def get_vector_manager():
    """Cached VectorManager factory."""
    try:
        load_dotenv()
        return VectorManager()
    except Exception:
        return None

def init_session_state():
    """Initializes globals and syncs data if needed."""
    if 'receipts_processed' not in st.session_state:
        st.session_state.receipts_processed = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = load_history()
    if 'vector_manager' not in st.session_state:
        st.session_state.vector_manager = get_vector_manager()
    if 'query_engine' not in st.session_state and st.session_state.vector_manager:
        st.session_state.query_engine = QueryEngine(st.session_state.vector_manager)
    
    if not st.session_state.receipts_processed:
        auto_sync_receipts()

def load_history():
    """Loads query history from disk."""
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            for item in history:
                if 'timestamp' in item:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                if isinstance(item.get('result'), dict):
                    # Robust wrapper for serialized history results
                    class Wrapper:
                        def __init__(self, d):
                            # Essential attributes for UI components
                            self.answer = d.get('answer', "No answer recorded.")
                            self.items = d.get('items', [])
                            self.receipts = d.get('receipts', [])
                            self.confidence = d.get('confidence', 0.0)
                            self.processing_time = d.get('processing_time', 0.0)
                            self.query_type = d.get('query_type', 'general')
                            # Map any other fields found in dict
                            for k, v in d.items():
                                if not hasattr(self, k): setattr(self, k, v)
                    item['result'] = Wrapper(item['result'])
            return history
    except Exception: return []

def save_history(history):
    """Serializes query history."""
    try:
        serialized = []
        for item in history:
            s_item = item.copy()
            if isinstance(item['timestamp'], datetime):
                s_item['timestamp'] = item['timestamp'].isoformat()
            if hasattr(item['result'], 'answer'):
                res = item['result']
                s_item['result'] = {
                    'answer': res.answer, 'confidence': res.confidence,
                    'receipts': res.receipts[:5], 'items': res.items[:5],
                    'processing_time': getattr(res, 'processing_time', 0.0)
                }
            serialized.append(s_item)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2)
    except Exception as e:
        logger.error(f"Save history failed: {e}")

def auto_sync_receipts():
    """Bootstraps local data into session state and vector DB."""
    from pathlib import Path
    receipt_dir = Path("data/receipt_samples_100")
    if not receipt_dir.exists(): return

    receipt_files = sorted(receipt_dir.glob("receipt_*.txt"))
    parser, chunker = ReceiptParser(), ReceiptChunker()
    all_receipts, all_chunks = [], []

    vm = st.session_state.vector_manager
    needs_indexing = True
    if vm:
        try:
            if vm.get_index_stats()['total_vector_count'] > 0: needs_indexing = False
        except Exception: pass

    with st.spinner("Syncing intelligence data..."):
        for f in receipt_files:
            try:
                content = f.read_text(encoding='utf-8')
                receipt = parser.parse_receipt(content, filename=f.name)
                all_receipts.append(receipt)
                if needs_indexing: all_chunks.extend(chunker.chunk_receipt(receipt))
            except Exception: continue

    st.session_state.receipts_processed.extend(all_receipts)
    if needs_indexing and all_chunks and vm:
        try: vm.index_chunks(all_chunks, batch_size=50)
        except Exception: pass

def apply_styles():
    """Injects premium Glassmorphism CSS."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');
            .main { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
            h1, h2, h3 { font-family: 'Outfit', sans-serif !important; background: linear-gradient(to right, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .query-container { background: rgba(99, 102, 241, 0.1); border-radius: 1rem; padding: 1.2rem; margin-bottom: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.2); }
            .query-label { font-size: 0.8rem; font-weight: 700; color: #a855f7; text-transform: uppercase; margin-bottom: 0.5rem; }
            .response-label { font-size: 0.8rem; font-weight: 700; color: #6366f1; text-transform: uppercase; margin-bottom: 0.5rem; }
        </style>
    """, unsafe_allow_html=True)

def render_ui():
    """Main UI layout logic."""
    st.set_page_config(page_title="Receipt Intelligence", page_icon="💸", layout="wide")
    apply_styles()
    
    st.title("💸 Intelligence Assistant")
    st.caption("Industrial RAG-powered Receipt Insights")

    # Sidebar Stats
    with st.sidebar:
        st.header("📊 Overview")
        if st.session_state.receipts_processed:
            count = len(st.session_state.receipts_processed)
            total = sum(r.total_amount for r in st.session_state.receipts_processed)
            st.metric("Total Receipts", count)
            st.metric("Total Spoken", f"${total:,.2f}")
        
        if st.button("🚀 Re-sync Data", type="secondary"):
            auto_sync_receipts()
            st.rerun()

    # Main Interaction
    tab_chat, tab_dash, tab_admin = st.tabs(["💬 Assistant", "📈 Insights", "⚙️ System"])
    
    with tab_chat:
        render_chat_view()
        
    with tab_dash:
        render_full_dashboard(st.session_state.receipts_processed)
        
    with tab_admin:
        render_admin_view()

def render_chat_view():
    """Renders the chat interface and suggestions."""
    if st.session_state.get('suggestion'):
        st.session_state.query_box = st.session_state.suggestion
        del st.session_state.suggestion

    q = st.text_input("Ask about your spending...", placeholder="e.g. Total spent at Walmart in Jan?", key="query_box")
    
    # Quick Suggestions
    cols = st.columns(4)
    suggestions = [
        ("📅 Jan 2024", "How much did I spend in January 2024?"),
        ("🛒 Last Week", "What did I buy last week?"),
        ("📄 Dec Receipts", "Show me all receipts from December"),
        ("🥬 Whole Foods", "Find all Whole Foods receipts"),
        ("☕ Coffee Shops", "How much have I spent at coffee shops?"),
        ("🍽️ Restaurants", "What's my total spending at restaurants?"),
        ("📱 Electronics", "Show me all electronics purchases"),
        ("🔒 Warranty", "Find receipts with warranty information"),
        ("💊 Pharmacy", "What pharmacy items did I buy?"),
        ("🛍️ Groceries > $5", "List all groceries over $5"),
        ("❤️ Health Items", "Find health-related purchases"),
        ("🍬 Treats", "Show me treats I bought"),
    ]
    for i, (label, val) in enumerate(suggestions):
        if cols[i % 4].button(label, type="secondary"):
            st.session_state.suggestion = val
            st.rerun()

    if q and q != st.session_state.get('last_q'):
        st.session_state.last_q = q
        engine = st.session_state.query_engine
        if engine:
            with st.spinner("Analyzing..."):
                res = engine.process_query(q)
                st.session_state.query_history.append({'query': q, 'timestamp': datetime.now(), 'result': res})
                save_history(st.session_state.query_history)
                st.rerun()

    for msg in reversed(st.session_state.query_history):
        render_response_feed_item(msg)

def render_admin_view():
    """System controls."""
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.query_history = []
        save_history([])
        st.rerun()
    
    if st.button("🛠️ Rebuild Vector Index", type="secondary"):
        if st.session_state.vector_manager:
            st.session_state.vector_manager.rebuild_index()
            st.success("Index rebuild triggered.")

if __name__ == "__main__":
    setup_logging("receipt_ui")
    init_session_state()
    render_ui()
