"""
Visualization components for query results and item distribution.
"""

import streamlit as st
import plotly.express as px
from typing import List, Dict

def render_item_visualization(items_data: List[Dict], key_prefix: str = "default"):
    """Render item visualizations with unique keys."""
    if not items_data:
        return
        
    st.markdown("#### 📊 Result Insights")
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Categories", "Merchants"])
    
    with tab1:
        prices = [float(item['Price'].replace('$', '').replace(',', '')) for item in items_data]
        fig = px.histogram(
            x=prices,
            nbins=10,
            title="Price Distribution",
            labels={'x': 'Price ($)', 'y': 'Count'},
            template="plotly_dark"
        )
        st.plotly_chart(fig, key=f"{key_prefix}_hist")
    
    with tab2:
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
        st.plotly_chart(fig, key=f"{key_prefix}_pie")
    
    with tab3:
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
        st.plotly_chart(fig, key=f"{key_prefix}_bar")

def render_response_feed_item(msg: Dict):
    """Renders a single chat history item with styled containers."""
    with st.chat_message("assistant"):
        # Query Box
        st.markdown(f"""
            <div class="query-container">
                <div class="query-label">🧑‍💻 Your Query</div>
                <div style="font-size: 1.1rem; font-weight: 500;">{msg['query']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Response Box
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
                    render_item_visualization(
                        items_data, 
                        key_prefix=f"history_{msg['timestamp'].strftime('%Y%m%d%H%M%S')}"
                    )
            
            st.caption(f"⚡ {msg['result'].processing_time:.2f}s | 🎯 {msg['result'].confidence:.1%} confidence | 📅 {msg['timestamp'].strftime('%b %d, %Y - %H:%M')}")
