"""
Financial Intelligence Dashboard component for the Receipt Intelligence System.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List

def render_financial_metrics(receipts: List):
    """Renders the top-level metrics bar."""
    if not receipts:
        return
    
    m1, m2, m3, m4 = st.columns(4)
    total_spent = sum(r.total_amount for r in receipts)
    total_items = sum(len(r.items) for r in receipts)
    avg_receipt = total_spent / len(receipts) if receipts else 0
    unique_merchants = len(set(r.merchant_name for r in receipts))
    
    m1.metric("Lifetime Spend", f"${total_spent:,.2f}")
    m2.metric("Total Items", f"{total_items:,}")
    m3.metric("Avg. Receipt", f"${avg_receipt:,.2f}")
    m4.metric("Active Merchants", unique_merchants)

def render_spending_velocity(receipts: List):
    """Renders spending velocity over time."""
    if not receipts:
        return
        
    df_time = pd.DataFrame([
        {'Date': r.transaction_date, 'Amount': float(r.total_amount)} 
        for r in receipts
    ]).sort_values('Date')
    
    fig = px.line(
        df_time, x='Date', y='Amount', 
        title='Spending Velocity Over Time',
        template="plotly_dark",
        color_discrete_sequence=['#a855f7']
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, key="dash_line")

def render_merchant_loyalty(receipts: List):
    """Renders top loyalty destinations."""
    if not receipts:
        return
        
    merchant_totals = {}
    for r in receipts:
        merchant_totals[r.merchant_name] = merchant_totals.get(r.merchant_name, 0) + float(r.total_amount)
    top_m = sorted(merchant_totals.items(), key=lambda x: x[1], reverse=True)[:8]
    
    fig = px.bar(
        x=[m[0] for m in top_m], y=[m[1] for m in top_m],
        title='Top Loyalty Destinations',
        template="plotly_dark",
        color_discrete_sequence=['#6366f1']
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, key="dash_bar")

def render_category_allocation(receipts: List):
    """Renders category spending breakdown."""
    if not receipts:
        return
        
    category_totals = {}
    for r in receipts:
        for item in r.items:
            cat = item.category.value if item.category else "Other"
            category_totals[cat] = category_totals.get(cat, 0) + float(item.total_price)
    
    fig = px.pie(
        values=list(category_totals.values()),
        names=list(category_totals.keys()),
        title='Category Allocation',
        hole=0.4,
        template="plotly_dark"
    )
    st.plotly_chart(fig, key="dash_pie")

def render_full_dashboard(receipts: List):
    """Orchestrates the full dashboard rendering."""
    if not receipts:
        st.info("No processing data available to generate insights.")
        return
    
    st.subheader("💡 Financial Intelligence Dashboard")
    render_financial_metrics(receipts)
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        render_spending_velocity(receipts)
    with c2:
        render_merchant_loyalty(receipts)
        
    st.markdown("---")
    render_category_allocation(receipts)
