"""
Answer generation logic for converting structured results into natural language.
"""

import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal
from openai import OpenAI

try:
    from ..utils.logging_config import logger
except ImportError:
    from utils.logging_config import logger


class AnswerGenerator:
    """
    Generates user-friendly natural language answers using LLMs.
    """
    
    def __init__(self, openai_client: OpenAI):
        """Initializes the generator with a pre-configured OpenAI client."""
        self.openai_client = openai_client

    def generate_answer(self, query: str, search_results: List[Dict[str, Any]], query_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for generating a grounded answer.
        Uses a structured context preparation and prompt generation flow.
        """
        query_params = query_params or {}
        
        # 1. Prepare Grounding Context
        context = self._prepare_context(query, search_results, query_params)
        
        # 2. Create Grounded Prompt
        prompt = self._create_prompt(context, query_params)
        
        # 3. Generate LLM Answer
        try:
            logger.debug(f"Generating LLM answer for query: {query}")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial assistant specialized in receipt data analysis. Be precise, concise, and helpful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0
            )
            answer = response.choices[0].message.content.strip()
            
            # Extract source filenames
            sources = list(set([r.get('filename') for r in context['receipts'] if r.get('filename')]))
            
            return {'answer': answer, 'sources': sources}
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            fallback = self._generate_fallback(context, query_params)
            return {'answer': fallback, 'sources': []}

    def _prepare_context(self, query: str, results: List[Dict[str, Any]], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Formats search results and metadata into a context dictionary."""
        unique_receipts = {}
        items = []
        for r in results:
            meta = r.get('metadata', {})
            rid = meta.get('receipt_id')
            if rid and rid not in unique_receipts:
                unique_receipts[rid] = meta
            if meta.get('chunk_type') == 'item_detail':
                items.append(meta)
                
        return {
            'query': query,
            'items_count': len(items),
            'receipts_count': len(unique_receipts),
            'receipts': list(unique_receipts.values()),
            'items': items[:15], # Limit sample size for context window
            'aggregations': query_params.get('audited_aggregation', {})
        }

    def _create_prompt(self, context: Dict[str, Any], query_params: Dict[str, Any]) -> str:
        """Constructs a detailed prompt with available grounding data."""
        prompt = f"""Question: "{context['query']}"
        
Grounding Data Found:
- Items Matching: {context['items_count']}
- Receipts Matching: {context['receipts_count']}
"""
        # Inject audited/verified totals strictly
        if context.get('aggregations') and 'value' in context['aggregations']:
            val = context['aggregations']['value']
            agg_type = query_params.get('aggregation', 'total')
            prompt += f"- VERIFIED {agg_type.upper()} (Audited): ${val:.2f}\n"

        if context['receipts']:
            prompt += "\nTop Relevant Receipts:\n"
            for r in context['receipts'][:5]:
                prompt += f"- {r.get('merchant_name')} ({r.get('transaction_date', 'N/A')}): ${r.get('total_amount', 0):.2f}"
                if r.get('payment_method'): prompt += f" via {r['payment_method']}"
                prompt += "\n"
                
        if context['items']:
            prompt += "\nSpecific Line Items:\n"
            for item in context['items']:
                prompt += f"- {item.get('name')} (${item.get('price', 0):.2f}) at {item.get('merchant')} on {item.get('date')} [{item.get('payment_method')}]\n"
                
        prompt += "\nInstructions: Provide a structured, professional response. If a VERIFIED sum/average is provided above, use that EXACT figure in your answer. Do not perform your own math if an audited figure is available."
        return prompt

    def _generate_fallback(self, context: Dict[str, Any], query_params: Dict[str, Any]) -> str:
        """Simple template-based answer for when LLM is unavailable."""
        count = context['receipts_count']
        if count == 0:
            return "Reviewing your files, I couldn't find any data matching that specific query."
            
        agg = context.get('aggregations')
        if agg and 'value' in agg:
            return f"I found {count} relevant receipts with a verified calculated value of ${agg['value']:.2f}."
            
        return f"I found {count} relevant receipts matching your request in your records."
