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

    def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for generating a grounded answer.
        
        Args:
            query: User's original natural language question.
            search_results: Chunks retrieved from the vector database.
            
        Returns:
            Dict containing the 'answer' string and 'sources' list.
        """
        # Prepare context by extracting useful content from results
        context_str = ""
        sources = []
        for i, res in enumerate(search_results[:10]): # Limit to top 10 for context window
            meta = res.get('metadata', {})
            content = meta.get('content', '')
            fname = meta.get('filename', 'Unknown')
            context_str += f"[{i+1}] {content} (Source: {fname})\n\n"
            if fname not in sources:
                sources.append(fname)

        prompt = f"""You are a professional financial assistant. 
Use the following receipt context to answer the user's question. 
If the answer isn't in the context, say you don't know based on the provided data.
DO NOT hallucinate facts or merchants not present in the context.

Context:
{context_str}

Question: {query}

Answer strictly based on the context above:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful receipt data assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content.strip()
            return {'answer': answer, 'sources': sources}
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return {'answer': f"I found the data but encountered an error generating a natural language response. (Found {len(search_results)} matching records)", 'sources': sources}

    def generate(self, query: str, results: Dict[str, Any], query_params: Dict[str, Any]) -> str:
        """
        Orchestrates prompt creation and LLM completion.
        """
        context = self._prepare_context(query, results)
        prompt = self._create_prompt(context, query_params)
        
        try:
            logger.debug(f"Sending prompt to OpenAI for query: {query}")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial assistant specialized in receipt data analysis. Be precise, concise, and helpful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0
            )
            answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated natural language answer.")
            return answer
        except Exception as e:
            logger.error(f"LLM Answer Generation failed: {e}")
            return self._generate_fallback(context, query_params)

    def _prepare_context(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Formats search results into a context dictionary for the prompting engine."""
        return {
            'query': query,
            'items_count': len(results.get('items', [])),
            'receipts_count': len(results.get('receipts', [])),
            'total_amount': results.get('total_amount', 0),
            'aggregations': results.get('aggregations', {}),
            'receipts': results.get('receipts', []),
            'items': results.get('items', [])[:10]  # Limit sample size
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
        count = context['items_count']
        total = context['total_amount']
        if count == 0:
            return "Reviewing your files, I couldn't find any data matching that specific query."
        return f"I found {count} relevant entries totaling ${total:.2f} in your account records."
