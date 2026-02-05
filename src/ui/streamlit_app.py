"""
Streamlit UI for receipt processing and querying system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from models import Receipt, ReceiptChunk
from parsers import ReceiptParser
from chunking import ReceiptChunker
from vectorstore import VectorManager
from query import QueryEngine
from utils.logging_config import logger, setup_logging

# Initialize logging for the UI
setup_logging("receipt_ui")


@st.cache_resource
def get_vector_manager():
    """Get or create cached VectorManager instance."""
    try:
        load_dotenv()
        return VectorManager()
    except Exception as e:
        return None

HISTORY_FILE = "data/query_history.json"

def save_history(history):
    """Save query history to a JSON file."""
    try:
        # Convert datetime to string for JSON serialization
        serializable_history = []
        for item in history:
            serializable_item = item.copy()
            if isinstance(item['timestamp'], datetime):
                serializable_item['timestamp'] = item['timestamp'].isoformat()
            
            # Extract basic info from result to avoid complex object issues
            if hasattr(item['result'], 'answer'):
                res = item['result']
                serializable_item['result'] = {
                    'answer': res.answer,
                    'confidence': res.confidence,
                    'receipts': res.receipts[:5], # Store only top 5 for size
                    'items': res.items[:5],
                    'processing_time': getattr(res, 'processing_time', 0.0),
                    'query_type': getattr(res, 'query_type', 'general')
                }
            serializable_history.append(serializable_item)
            
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving history: {e}")

def load_history():
    """Load query history from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            # Reconstruct datetime and dummy result object
            for item in history:
                if 'timestamp' in item:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                # Reconstruct result-like object for rendering
                if isinstance(item.get('result'), dict):
                    # We just need it to have .answer and .confidence for the UI
                    class DummyResult:
                        def __init__(self, d):
                            self.answer = d.get('answer', '')
                            self.confidence = d.get('confidence', 0.0)
                            self.receipts = d.get('receipts', [])
                            self.items = d.get('items', [])
                            self.processing_time = d.get('processing_time', 0.0)
                            self.query_type = d.get('query_type', 'general')
                    item['result'] = DummyResult(item['result'])
            return history
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return []

def init_session_state():
    """Initialize session state variables."""
    # Initialize lists first to prevent AttributeErrors
    if 'receipts_processed' not in st.session_state:
        st.session_state.receipts_processed = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = load_history()

    if 'vector_manager' not in st.session_state:
        st.session_state.vector_manager = get_vector_manager()
        if st.session_state.vector_manager:
            try:
                st.session_state.query_engine = QueryEngine(st.session_state.vector_manager)
            except:
                st.session_state.query_engine = None
    
    if 'query_engine' not in st.session_state and st.session_state.vector_manager:
        st.session_state.query_engine = QueryEngine(st.session_state.vector_manager)
    
    # Auto-sync receipts for dashboard if empty
    if not st.session_state.receipts_processed:
        auto_process_receipts_from_folder()


def apply_custom_styles():
    """Apply premium custom CSS to the Streamlit app."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;800&display=swap');
            
            :root {
                --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                --glass-bg: rgba(255, 255, 255, 0.03);
                --glass-border: rgba(255, 255, 255, 0.1);
            }

            /* Main Layout */
            .main {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Inter', sans-serif;
            }
            
            /* Typography */
            h1, h2, h3 {
                font-family: 'Outfit', sans-serif !important;
                background: linear-gradient(to right, #6366f1, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800 !important;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #020617;
                border-right: 1px solid var(--glass-border);
            }
            
            /* Metric Cards */
            [data-testid="stMetric"] {
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                padding: 1rem;
                border-radius: 1rem;
                backdrop-filter: blur(12px);
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
                background-color: transparent;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 4px 4px 0 0;
                gap: 1rem;
                padding-top: 10px;
                padding-bottom: 10px;
                color: #94a3b8;
            }
            .stTabs [aria-selected="true"] {
                background-color: transparent !important;
                color: #a855f7 !important;
                border-bottom: 2px solid #a855f7 !important;
            }

            /* Chat Messages */
            [data-testid="stChatMessage"] {
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                border-radius: 1.5rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                backdrop-filter: blur(8px);
            }
            
            /* Success/Info boxes */
            .stAlert {
                background: var(--glass-bg) !important;
                border: 1px solid var(--glass-border) !important;
                color: #e2e8f0 !important;
                border-radius: 1rem !important;
            }
            
            /* Nested Query Box */
            .query-container {
                background: rgba(99, 102, 241, 0.1);
                border-radius: 1rem;
                padding: 1.2rem;
                margin-bottom: 1.5rem;
                border: 1px solid rgba(99, 102, 241, 0.2);
            }
            .query-label {
                font-size: 0.8rem;
                font-weight: 700;
                color: #a855f7;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }
            
            .response-container {
                background: rgba(255, 255, 255, 0.02);
                border-radius: 1rem;
                padding: 1.2rem;
                border: 1px solid var(--glass-border);
            }
            .response-label {
                font-size: 0.8rem;
                font-weight: 700;
                color: #6366f1;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }

            /* Hide Streamlit elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Receipt Intelligence",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_styles()


def auto_process_receipts_from_folder():
    """Auto-process receipts from the data folder on initialization."""
    from pathlib import Path
    
    if not st.session_state.vector_manager:
        logger.warning("Vector Manager not available for auto-processing")
        return

    # Check if session state is already populated
    if st.session_state.receipts_processed:
        logger.info("Session state already populated. Skipping sync.")
        return

    # 1. Check if we need to index (Pinecone)
    needs_indexing = True
    try:
        stats = st.session_state.vector_manager.get_index_stats()
        if stats['total_vector_count'] > 0:
            needs_indexing = False
            logger.info("Data already indexed in Pinecone. Skipping indexing step.")
    except Exception as e:
        logger.warning(f"Could not check index stats: {e}")
    
    # 2. Always parse local files to populate Dashboard state
    receipt_dir = Path("data/receipt_samples_100")
    if not receipt_dir.exists():
        logger.error(f"Receipt directory not found: {receipt_dir}")
        return
    
    receipt_files = sorted(receipt_dir.glob("receipt_*.txt"))
    if not receipt_files:
        logger.warning("No receipt files found in sample directory")
        return
    
    logger.info(f"Syncing {len(receipt_files)} receipts for visual dashboard...")
    
    parser = ReceiptParser()
    chunker = ReceiptChunker()
    
    all_receipts = []
    all_chunks = []
    
    # Process files
    with st.spinner("Syncing dashboard data..."):
        for i, file_path in enumerate(receipt_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                receipt = parser.parse_receipt(content, filename=file_path.name)
                all_receipts.append(receipt)
                
                if needs_indexing:
                    chunks = chunker.chunk_receipt(receipt)
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                continue
    
    # Update Session State (Dashboard)
    st.session_state.receipts_processed.extend(all_receipts)
    logger.info(f"Synced {len(all_receipts)} receipts to session state.")

    # 3. Index only if needed
    if needs_indexing and all_chunks:
        try:
            with st.spinner(f"Indexing {len(all_chunks)} chunks to Pinecone..."):
                st.session_state.vector_manager.index_chunks(all_chunks, batch_size=50)
            logger.info(f"Successfully indexed {len(all_chunks)} chunks.")
            st.success(f"Initialized system with {len(all_receipts)} receipts!")
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            st.error(f"Sync complete for dashboard, but indexing failed: {e}")
    elif not needs_indexing:
        st.success(f"Dashboard synced with {len(all_receipts)} local receipts. Vector index is ready.")

    # Index chunks
    if all_chunks:
        try:
            print("📤 Indexing chunks to Pinecone...")
            with st.spinner(f"Indexing {len(all_chunks)} chunks..."):
                st.session_state.vector_manager.index_chunks(all_chunks, batch_size=50)
                st.session_state.receipts_processed.extend(all_receipts)
            print(" Indexing complete!")
            st.success(f" Successfully indexed {len(all_chunks)} chunks from {len(all_receipts)} receipts!")
        except Exception as e:
            print(f" Error indexing chunks: {e}")
            st.error(f"Error indexing chunks: {e}")


def render_sidebar():
    """Render the sidebar with controls and information."""
    st.sidebar.title(" Receipt Intelligence")
    
    # System Status
    st.sidebar.subheader("System Status")
    
    if st.session_state.vector_manager:
        try:
            stats = st.session_state.vector_manager.get_index_stats()
            st.sidebar.success(f" Connected to Vector DB")
            st.sidebar.info(f" {stats['total_vector_count']} chunks indexed")
        except Exception as e:
            st.sidebar.error(f" Vector DB Error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Vector DB not connected")
        if hasattr(st.session_state, 'init_error'):
            st.sidebar.error(f"Error: {st.session_state.init_error[:100]}")
            st.sidebar.info(" Use Admin tab to manually initialize")
    
    # Quick Stats
    if st.session_state.receipts_processed:
        st.sidebar.subheader("Quick Stats")
        receipts_count = len(st.session_state.receipts_processed)
        total_items = sum(len(receipt.items) for receipt in st.session_state.receipts_processed)
        total_spent = sum(receipt.total_amount for receipt in st.session_state.receipts_processed)
        
        st.sidebar.metric("Receipts", receipts_count)
        st.sidebar.metric("Items", total_items)
        st.sidebar.metric("Total Spent", f"${total_spent:.2f}")
    
    # Processing Controls
    st.sidebar.divider()
    st.sidebar.subheader("🔄 Data Management")
    
    if st.sidebar.button("🚀 Process Data (Scrape)", type="primary"):
        with st.spinner("Processing receipts data..."):
            auto_process_receipts_from_folder()
            st.success("Processing complete!")
            st.rerun()

    # Sample Queries moved to Main View as requested
    # Removed from Sidebar


def render_header():
    """Render the main header."""
    st.title("💸 Intelligence Assistant")
    st.caption("Industrial RAG-powered Receipt Insights")


def render_quick_starts():
    """Render a high-density grid of quick-start query buttons."""
    st.markdown("### ⚡ Quick Insights")
    
    queries = [
        ("📅 Jan Spending", "How much did I spend in January 2024?"),
        ("🛒 Groceries > $5", "List all groceries over $5"),
        ("📱 Electronics", "Show me all electronics purchases"),
        ("💊 Pharmacy", "What pharmacy items did I buy?"),
        ("🏷️ Walmart", "Find all Walmart receipts"),
        ("🏥 Health Care", "Find health-related purchases"),
        ("☕ Coffee Shops", "How much spent at coffee shops?"),
        ("🍕 Restaurants", "Total spending at restaurants"),
        ("🛠️ Supplies", "How much did I spend on supplies?"),
        ("🧼 Personal Care", "Spent on personal care items?"),
        ("🍭 Treats", "Show me treats I bought")
    ]
    
    # Create 4 columns for buttons (apx 11 items row-wise)
    cols = st.columns(4)
    for i, (label, query) in enumerate(queries):
        if cols[i % 4].button(label, use_container_width=True, key=f"btn_quick_{i}"):
            st.session_state.current_query = query
            st.rerun()


# Removed process_uploaded_files as requested


def render_query_section():
    """Render the enhanced query interface."""
    
    # 1. Input Section at the top
    st.markdown("### 🔎 Search Receipts")
    query_col, clear_col = st.columns([5, 1])
    
    with query_col:
        query = st.text_input(
            "What would you like to know?",
            placeholder="e.g., How much did I spend at Walmart last month?",
            label_visibility="collapsed",
            key="query_input_main"
        )
    
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.query_history = []
            st.rerun()

    # 2. Row-wise Suggestions directly below
    render_quick_starts()
    st.markdown("---")

    # 3. Handle processing
    if query and query != st.session_state.get('last_query', ''):
        st.session_state.last_query = query
        execute_query(query)
        st.rerun()

    # Handle sample query from trigger
    query_trigger = st.session_state.get('current_query', None)
    if query_trigger:
        st.session_state.last_query = query_trigger
        execute_query(query_trigger)
        del st.session_state.current_query
        st.rerun()

    # 4. Results / Chat History Section
    if st.session_state.get('query_history'):
        st.markdown("### 📜 Response Feed")
        for msg in reversed(st.session_state.get('query_history', [])):
            with st.chat_message("assistant"):
                # Nested Query Box
                st.markdown(f"""
                    <div class="query-container">
                        <div class="query-label">🧑‍💻 Your Query</div>
                        <div style="font-size: 1.1rem; font-weight: 500;">{msg['query']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Nested AI Response Box
                st.markdown('<div class="response-label">🤖 Intelligence Response</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(msg['result'].answer)
                    
                    # Inline visuals if helpful
                    if msg['result'].items and len(msg['result'].items) > 1:
                        with st.expander("📊 Data Visualizations", expanded=False):
                            items_data = []
                            for item in msg['result'].items:
                                items_data.append({
                                    'Price': f"${item.get('price', 0):.2f}",
                                    'Category': item.get('category', 'other'),
                                    'Merchant': item.get('merchant', 'Unknown'),
                                })
                            # Use a unique key for each chart in history
                            render_item_visualization(items_data, key_prefix=f"history_{msg['timestamp'].strftime('%Y%m%d%H%M%S')}")
                    
                    st.caption(f"⚡ {msg['result'].processing_time:.2f}s | 🎯 {msg['result'].confidence:.1%} confidence | 📅 {msg['timestamp'].strftime('%b %d, %Y - %H:%M')}")


def execute_query(query: str):
    """Execute a query and display results."""
    if not st.session_state.query_engine or not query:
        return
    
    with st.spinner("Analyzing receipt data..."):
        start_time = time.time()
        result = st.session_state.query_engine.process_query(query)
        processing_time = time.time() - start_time
    
    # Add to query history
    st.session_state.query_history.append({
        'query': query,
        'timestamp': datetime.now(),
        'result': result
    })
    save_history(st.session_state.query_history)


# Removed render_query_results as requested


def render_item_visualization(items_data: List[Dict], key_prefix: str = "default"):
    """Render item visualizations with unique keys."""
    st.markdown("####  Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Categories", "Merchants"])
    
    with tab1:
        # Price distribution
        prices = [float(item['Price'].replace('$', '').replace(',', '')) for item in items_data]
        fig = px.histogram(
            x=prices,
            nbins=10,
            title="Price Distribution",
            labels={'x': 'Price ($)', 'y': 'Count'},
            template="plotly_dark"
        )
        st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_hist")
    
    with tab2:
        # Category breakdown
        category_counts = {}
        for item in items_data:
            cat = item['Category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        fig = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="Items by Category",
            template="plotly_dark"
        )
        st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_pie")
    
    with tab3:
        # Merchant breakdown
        merchant_counts = {}
        for item in items_data:
            merchant = item['Merchant']
            merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1
        
        fig = px.bar(
            x=list(merchant_counts.keys()),
            y=list(merchant_counts.values()),
            title="Items by Merchant",
            labels={'x': 'Merchant', 'y': 'Count'},
            template="plotly_dark"
        )
        st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_bar")


# Removed render_history_section as requested


def render_financial_dashboard():
    """Render a sleek financial oversight dashboard."""
    if not st.session_state.receipts_processed:
        st.info("No processing data available to generate insights.")
        return
    
    st.subheader("💡 Financial Intelligence Dashboard")
    
    receipts = st.session_state.receipts_processed
    
    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    total_spent = sum(r.total_amount for r in receipts)
    total_items = sum(len(r.items) for r in receipts)
    avg_receipt = total_spent / len(receipts) if receipts else 0
    unique_merchants = len(set(r.merchant_name for r in receipts))
    
    m1.metric("Lifetime Spend", f"${total_spent:,.2f}")
    m2.metric("Total Items", f"{total_items:,}")
    m3.metric("Avg. Receipt", f"${avg_receipt:,.2f}")
    m4.metric("Active Merchants", unique_merchants)

    st.markdown("---")
    
    # Visuals
    c1, c2 = st.columns(2)
    
    # Prepare data
    df_time = pd.DataFrame([{'Date': r.transaction_date, 'Amount': float(r.total_amount)} for r in receipts]).sort_values('Date')
    
    with c1:
        fig_line = px.line(
            df_time, x='Date', y='Amount', 
            title='Spending Velocity Over Time',
            template="plotly_dark",
            color_discrete_sequence=['#a855f7']
        )
        fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, width='stretch', key="dash_line")
    
    with c2:
        merchant_totals = {}
        for r in receipts:
            merchant_totals[r.merchant_name] = merchant_totals.get(r.merchant_name, 0) + float(r.total_amount)
        top_m = sorted(merchant_totals.items(), key=lambda x: x[1], reverse=True)[:8]
        
        fig_bar = px.bar(
            x=[m[0] for m in top_m], y=[m[1] for m in top_m],
            title='Top Loyalty Destinations',
            template="plotly_dark",
            color_discrete_sequence=['#6366f1']
        )
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, width='stretch', key="dash_bar")

    # Categories
    category_totals = {}
    for r in receipts:
        for item in r.items:
            cat = item.category.value if item.category else "Other"
            category_totals[cat] = category_totals.get(cat, 0) + float(item.total_price)
    
    fig_pie = px.pie(
        values=list(category_totals.values()),
        names=list(category_totals.keys()),
        title='Category Allocation',
        hole=0.4,
        template="plotly_dark"
    )
    st.plotly_chart(fig_pie, width='stretch', key="dash_pie")


def render_admin_section():
    """Render admin controls."""
    st.subheader("⚙️ System Administration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Initialize Vector DB", type="secondary"):
            with st.spinner("Initializing vector database..."):
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    st.session_state.vector_manager = VectorManager()
                    st.session_state.query_engine = QueryEngine(st.session_state.vector_manager)
                    st.success(" Vector database initialized successfully!")
                except Exception as e:
                    st.error(f" Error initializing vector database: {str(e)}")
    
    with col2:
        if st.button(" Clear All Data", type="secondary"):
            if st.session_state.vector_manager:
                try:
                    st.session_state.vector_manager.rebuild_index()
                    st.session_state.receipts_processed = []
                    st.session_state.query_history = []
                    st.success(" All data cleared!")
                except Exception as e:
                    st.error(f" Error clearing data: {str(e)}")
    
    # System Statistics
    if st.session_state.vector_manager:
        try:
            stats = st.session_state.vector_manager.get_index_stats()
            st.markdown("###  System Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Vectors", stats['total_vector_count'])
            
            with col2:
                st.metric("Index Fullness", f"{stats['index_fullness']:.2%}")
            
            with col3:
                st.metric("Dimensions", stats['dimension'])
        except Exception as e:
            st.error(f"Error getting stats: {str(e)}")


def main():
    """Main application entry point."""
    setup_page_config()
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Assistant", "⚙️ Control Center"])
    
    with tab1:
        render_query_section()
    
    with tab2:
        render_admin_section()
        st.markdown("---")
        render_financial_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Receipt Intelligence System**  
    🚀 Powered by vector embeddings and natural language processing
    """)


if __name__ == "__main__":
    main()
