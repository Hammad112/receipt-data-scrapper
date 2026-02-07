"""
Response synthesis module for the Receipt RAG system.

This module converts retrieved receipt data and verified audit results 
into concise, natural language answers grounded in source evidence.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.utils.logging_config import logger


class AnswerGenerator:
    """
    Synthesizes natural language answers from retrieved context.
    
    Responsibilities:
    - Formatting retrieved chunks for LLM consumption.
    - Injecting deterministic audit results to prevent hallucinations.
    - Enforcing strict grounding in the provided context.
    """

    SYSTEM_PROMPT = """You are a precise Receipt Intelligence Assistant.
            Your goal is to answer questions about the user's receipts using ONLY the provided context.

            RULES:
            1. ALWAYS answer based on the provided context if results are present - do not say you don't know.
            2. If an 'audit_result' is provided, use its 'value' as the definitive answer for numerical queries.
            3. Ground all claims in the context. Mention specific merchants, dates, and items found.
            4. If the context is truly empty or says "NO RESULTS", then say you don't know based on the current receipts.
            5. For temporal queries like "last week" or "last month", the context already contains the filtered results for that period.
            6. Keep answers concise and professional.
            7. Use markdown for lists or emphasis where appropriate."""

    def __init__(self, model: str = "gpt-4o"):
        """Initializes the generator with a specific OpenAI model."""
        self.client = OpenAI()
        self.model = model

    def generate(
        self, 
        query: str, 
        context: List[Any], 
        query_params: Dict[str, Any], 
        audit_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a natural language answer based on context and audit data.
        
        Args:
            query: The original user question.
            context: List of retrieved receipt chunks metadata.
            query_params: Extracted parameters from the parser.
            audit_result: Deterministic calculation results for verification.
            
        Returns:
            A string containing the synthesized answer.
        """
        try:
            formatted_context = self._prepare_context(context)
            user_prompt = self._build_user_prompt(query, formatted_context, audit_result)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0, # Deterministic response
                max_tokens=500
            )

            answer = response.choices[0].message.content
            
            # Append verification badge if audit was successful
            if audit_result and audit_result.get('verified'):
                answer += "\n\n✅ *Verified against source receipts.*"
                
            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "I encountered an error while synthesizing the answer. Please try again."

    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a stable string for the LLM."""
        formatted = []
        for i, item in enumerate(context):
            meta = item.get('metadata', {})
            content = meta.get('content', '')
            formatted.append(f"--- Context {i+1} ---\n{content}")
        return "\n\n".join(formatted)

    def _build_user_prompt(self, query: str, context: str, audit_result: Optional[Dict[str, Any]]) -> str:
        """Constructs the user message with query, context and optional audit data."""
        prompt = f"Question: {query}\n\nContext:\n{context}\n\n"
        if audit_result:
            prompt += f"Note: A deterministic audit has been performed. Use this verified value if applicable: {audit_result}\n"
        prompt += "\nAnswer:"
        return prompt
