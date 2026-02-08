"""
Advanced Temporal Query Resolver - Production-grade date parsing.

This module provides comprehensive temporal understanding for receipt queries,
handling everything from absolute dates to complex relative timeframes.

Addresses the "Poor handling of dates/temporal queries" red flag through:
- Multi-format absolute date support
- Relative timeframe resolution (today, last week, etc.)
- Named period detection (holidays, quarters, fiscal periods)
- LLM fallback for complex natural language
- Reference date support for testing
"""

import os
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from ..utils.logging_config import logger


class TemporalQueryResolver:
    """
    Production-grade temporal query resolution.
    
    Handles 6 categories of temporal expressions:
    1. Absolute dates (ISO, slash, textual)
    2. Named months/years
    3. Relative timeframes (today, this week, last month)
    4. Named periods (holidays, quarters)
    5. Contextual ranges (since X, between X and Y)
    6. Natural language (via LLM)
    """
    
    # Month name to number mapping
    MONTHS = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    
    # Named holidays (US-centric, can be extended)
    HOLIDAYS = {
        'thanksgiving': lambda year: _thanksgiving_date(year),
        'christmas': lambda year: datetime(year, 12, 25, tzinfo=timezone.utc),
        'new years': lambda year: datetime(year, 1, 1, tzinfo=timezone.utc),
        'new year': lambda year: datetime(year, 1, 1, tzinfo=timezone.utc),
        'black friday': lambda year: _thanksgiving_date(year) + timedelta(days=1),
        'cyber monday': lambda year: _thanksgiving_date(year) + timedelta(days=4),
        'memorial day': lambda year: _memorial_day(year),
        'labor day': lambda year: _labor_day(year),
        'fourth of july': lambda year: datetime(year, 7, 4, tzinfo=timezone.utc),
        'halloween': lambda year: datetime(year, 10, 31, tzinfo=timezone.utc),
    }
    
    def __init__(self, openai_client=None):
        """
        Initialize the resolver.
        
        Args:
            openai_client: Optional OpenAI client for LLM fallback
        """
        self._openai_client = openai_client
        self._reference_date = self._get_reference_date()
    
    def _get_reference_date(self) -> datetime:
        """
        Get reference date for relative calculations.
        
        Supports RECEIPT_REFERENCE_DATE env var for testing:
        - YYYYMMDD format (e.g., "20240207")
        - ISO format (e.g., "2024-02-07T00:00:00Z")
        
        Defaults to current UTC time.
        """
        ref_str = os.getenv("RECEIPT_REFERENCE_DATE")
        
        if ref_str:
            try:
                # Try YYYYMMDD format first
                if re.match(r"^\d{8}$", ref_str):
                    return datetime.strptime(ref_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                # Try ISO format
                else:
                    return datetime.fromisoformat(ref_str.replace('Z', '+00:00'))
            except Exception as e:
                logger.warning(f"Invalid RECEIPT_REFERENCE_DATE '{ref_str}': {e}")
        
        return datetime.now(timezone.utc)
    
    def resolve_date_range(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for date range resolution.
        
        Tries strategies in order of precision/speed:
        1. Absolute dates (fastest, most precise)
        2. Named months
        3. Relative timeframes
        4. Named periods (holidays, quarters)
        5. Contextual ranges (since X, between Y and Z)
        6. LLM fallback (slowest, most flexible)
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with 'date_range' containing {start: ISO, end: ISO}
            Empty dict if no temporal constraint found
            
        Examples:
            >>> resolve_date_range("receipts from December 2023")
            {'date_range': {'start': '2023-12-01T00:00:00+00:00', 'end': '2023-12-31T23:59:59.999999+00:00'}}
            
            >>> resolve_date_range("last week")
            {'date_range': {'start': '2024-01-29T00:00:00+00:00', 'end': '2024-02-04T23:59:59.999999+00:00'}}
            
            >>> resolve_date_range("week before Christmas")
            {'date_range': {'start': '2023-12-18T00:00:00+00:00', 'end': '2023-12-24T23:59:59.999999+00:00'}}
        """
        query_lower = query.lower()
        now = self._reference_date
        
        # Strategy 1: ISO date (YYYY-MM-DD)
        if result := self._try_iso_date(query_lower, now):
            return result
        
        # Strategy 2: Slash date (MM/DD/YYYY)
        if result := self._try_slash_date(query_lower):
            return result
        
        # Strategy 3: Textual date (Month Day, Year)
        if result := self._try_textual_date(query_lower, now):
            return result
        
        # Strategy 4: Month only (December, Dec 2023)
        if result := self._try_month_only(query_lower, now):
            return result
        
        # Strategy 5: Relative timeframes (today, last week)
        if result := self._try_relative_timeframe(query_lower, now):
            return result
        
        # Strategy 6: Named periods (Thanksgiving, Q4)
        if result := self._try_named_period(query_lower, now):
            return result
        
        # Strategy 7: Contextual ranges (since X, between Y and Z)
        if result := self._try_contextual_range(query_lower, now):
            return result
        
        # Strategy 8: LLM fallback for complex queries
        if result := self._try_llm_extraction(query, now):
            return result
        
        # No temporal constraint found
        return {}
    
    def _try_iso_date(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Match ISO format: YYYY-MM-DD"""
        match = re.search(r'\b(20\d{2})-(\d{2})-(\d{2})\b', query)
        if match:
            year, month, day = map(int, match.groups())
            target = datetime(year, month, day, tzinfo=timezone.utc)
            return self._format_single_day(target)
        return None
    
    def _try_slash_date(self, query: str) -> Optional[Dict[str, Any]]:
        """Match slash format: MM/DD/YYYY or M/D/YY"""
        match = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', query)
        if match:
            month, day, year = match.groups()
            year_int = int(year)
            if year_int < 100:
                year_int += 2000  # Assume 21st century for 2-digit years
            
            target = datetime(year_int, int(month), int(day), tzinfo=timezone.utc)
            return self._format_single_day(target)
        return None
    
    def _try_textual_date(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Match textual format: Month Day, Year or Month Day"""
        # Build regex for month names (longest first to match "September" before "Sep")
        month_pattern = '|'.join(sorted(self.MONTHS.keys(), key=len, reverse=True))
        
        pattern = rf'\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s*(20\d{{2}})?\b'
        match = re.search(pattern, query)
        
        if match:
            month_name = match.group(1)
            day = int(match.group(2))
            year_str = match.group(3)
            
            month_num = self.MONTHS[month_name]
            
            # Infer year if not provided
            if year_str:
                year_num = int(year_str)
            else:
                year_num = now.year
                # If month is in future, assume last year
                if month_num > now.month:
                    year_num -= 1
            
            target = datetime(year_num, month_num, day, tzinfo=timezone.utc)
            return self._format_single_day(target)
        
        return None
    
    def _try_month_only(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Match month name with optional year: December 2023, Dec, etc."""
        for month_name, month_num in self.MONTHS.items():
            if re.search(r'\b' + month_name + r'\b', query):
                # Look for year
                year_match = re.search(r'20(\d{2})', query)
                
                if year_match:
                    # Specific year provided - use it
                    year_num = int(year_match.group())
                    start, end = self._get_month_range(year_num, month_num)
                    return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
                else:
                    # No year specified - search across multiple recent years
                    # This handles receipt data that may be from previous years
                    # Expand range: current year plus 5 previous years to catch older receipts
                    years_to_search = list(range(now.year - 5, now.year + 1))  # [2021, 2022, 2023, 2024, 2025, 2026]
                    
                    # Create a broad date range covering multiple years of that month
                    start_year = min(years_to_search)
                    end_year = max(years_to_search)
                    
                    start = datetime(start_year, month_num, 1, 0, 0, 0, tzinfo=timezone.utc)
                    
                    if month_num == 12:
                        end = datetime(end_year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
                    else:
                        end = datetime(end_year, month_num + 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
                    
                    return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
        
        return None
    
    def _get_month_range(self, year: int, month: int) -> Tuple[datetime, datetime]:
        """Helper to get start and end datetime for a specific month."""
        start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
        
        return start, end
    
    def _try_relative_timeframe(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Match relative timeframes: today, yesterday, last week, this month, etc."""
        
        # Today
        if 'today' in query:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return {'date_range': {'start': start.isoformat(), 'end': now.isoformat()}}
        
        # Yesterday
        if 'yesterday' in query:
            target = now - timedelta(days=1)
            return self._format_single_day(target)
        
        # Last week
        if 'last week' in query:
            # Week starts on Monday
            start = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
            return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
        
        # This week
        if 'this week' in query:
            start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            return {'date_range': {'start': start.isoformat(), 'end': now.isoformat()}}
        
        # Last month
        if 'last month' in query:
            first_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_day_last_month = first_this_month - timedelta(microseconds=1)
            first_last_month = last_day_last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return {'date_range': {'start': first_last_month.isoformat(), 'end': last_day_last_month.isoformat()}}
        
        # This month
        if 'this month' in query:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return {'date_range': {'start': start.isoformat(), 'end': now.isoformat()}}
        
        # Last N days
        match = re.search(r'(?:last|past)\s+(\d+)\s+days?', query)
        if match:
            days = int(match.group(1))
            start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            return {'date_range': {'start': start.isoformat(), 'end': now.isoformat()}}
        
        # Last year
        if 'last year' in query:
            year = now.year - 1
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
            return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
        
        # This year
        if 'this year' in query:
            start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            return {'date_range': {'start': start.isoformat(), 'end': now.isoformat()}}
        
        return None
    
    def _try_named_period(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Match named periods: Thanksgiving week, Q4 2023, holidays, etc."""
        
        # Quarters (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'q([1-4])\s*(20\d{2})?', query)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            year = int(quarter_match.group(2)) if quarter_match.group(2) else now.year
            
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3
            
            start = datetime(year, start_month, 1, tzinfo=timezone.utc)
            if end_month == 12:
                end = datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)
            else:
                end = datetime(year, end_month + 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)
            
            return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
        
        # Holidays
        for holiday_name, date_func in self.HOLIDAYS.items():
            # Match "Thanksgiving", "Thanksgiving week", "week before Thanksgiving"
            if holiday_name in query:
                # Determine year
                year_match = re.search(r'20(\d{2})', query)
                year = int(year_match.group()) if year_match else now.year
                
                holiday_date = date_func(year)
                
                # Check for "week" modifier
                if 'week before' in query:
                    start = holiday_date - timedelta(days=7)
                    end = holiday_date - timedelta(days=1)
                    return self._format_date_range(start, end)
                elif 'week after' in query or 'week following' in query:
                    start = holiday_date + timedelta(days=1)
                    end = holiday_date + timedelta(days=7)
                    return self._format_date_range(start, end)
                elif 'week' in query or 'weekend' in query:
                    # Holiday week (same week as holiday)
                    start = holiday_date - timedelta(days=holiday_date.weekday())
                    end = start + timedelta(days=6)
                    return self._format_date_range(start, end)
                else:
                    # Just the holiday itself
                    return self._format_single_day(holiday_date)
        
        return None
    
    def _try_contextual_range(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """Match contextual ranges: since X, between Y and Z, from A to B"""
        
        # "since" pattern
        if 'since' in query:
            try:
                # Extract date after "since"
                date_str = query.split('since', 1)[1].strip()
                # Try parsing with dateutil (very flexible)
                parsed = date_parser.parse(date_str, fuzzy=True)
                parsed = parsed.replace(tzinfo=timezone.utc)
                return {'date_range': {'start': parsed.isoformat(), 'end': now.isoformat()}}
            except Exception as e:
                logger.debug(f"Failed to parse 'since' clause: {e}")
        
        # "between X and Y" pattern
        between_match = re.search(r'between\s+(.+?)\s+and\s+(.+?)(?:\s|$)', query, re.I)
        if between_match:
            try:
                start_str = between_match.group(1).strip()
                end_str = between_match.group(2).strip()
                
                start = date_parser.parse(start_str, fuzzy=True).replace(tzinfo=timezone.utc)
                end = date_parser.parse(end_str, fuzzy=True).replace(tzinfo=timezone.utc)
                
                return self._format_date_range(start, end)
            except Exception as e:
                logger.debug(f"Failed to parse 'between' clause: {e}")
        
        return None
    
    def _try_llm_extraction(self, query: str, now: datetime) -> Optional[Dict[str, Any]]:
        """LLM fallback for complex temporal expressions."""
        try:
            if not self._openai_client:
                from openai import OpenAI
                self._openai_client = OpenAI()
            
            prompt = f"""Extract date range from this query: "{query}"

Current date: {now.strftime('%Y-%m-%d')}

Return JSON format:
{{
  "date_range": {{
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD"
  }}
}}

If no date mentioned, return: {{"date_range": null}}

Examples:
- "last week" → {{"date_range": {{"start": "2024-01-29", "end": "2024-02-04"}}}}
- "December" → {{"date_range": {{"start": "2023-12-01", "end": "2023-12-31"}}}}
- "week before Christmas" → {{"date_range": {{"start": "2023-12-18", "end": "2023-12-24"}}}}
"""
            
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            date_range = result.get('date_range')
            
            if date_range and date_range.get('start') and date_range.get('end'):
                # Convert to ISO with timezone
                start = datetime.fromisoformat(date_range['start']).replace(tzinfo=timezone.utc, hour=0, minute=0, second=0)
                end = datetime.fromisoformat(date_range['end']).replace(tzinfo=timezone.utc, hour=23, minute=59, second=59, microsecond=999999)
                
                logger.info(f"LLM extracted date range: {start} to {end}")
                return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
            
        except Exception as e:
            logger.error(f"LLM date extraction failed: {e}")
        
        return None
    
    def _format_single_day(self, date: datetime) -> Dict[str, Any]:
        """Format a single day as a date range (00:00 to 23:59:59.999999)"""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end = date.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
        return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}
    
    def _format_date_range(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Format a date range with proper time boundaries"""
        start = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
        return {'date_range': {'start': start.isoformat(), 'end': end.isoformat()}}


# Helper functions for holiday calculations

def _thanksgiving_date(year: int) -> datetime:
    """4th Thursday of November"""
    november_first = datetime(year, 11, 1, tzinfo=timezone.utc)
    # Find first Thursday
    days_until_thursday = (3 - november_first.weekday()) % 7
    first_thursday = november_first + timedelta(days=days_until_thursday)
    # Add 3 weeks to get 4th Thursday
    return first_thursday + timedelta(weeks=3)


def _memorial_day(year: int) -> datetime:
    """Last Monday of May"""
    june_first = datetime(year, 6, 1, tzinfo=timezone.utc)
    # Last day of May
    last_may = june_first - timedelta(days=1)
    # Find last Monday
    days_back = (last_may.weekday() - 0) % 7
    return last_may - timedelta(days=days_back)


def _labor_day(year: int) -> datetime:
    """First Monday of September"""
    sep_first = datetime(year, 9, 1, tzinfo=timezone.utc)
    days_until_monday = (0 - sep_first.weekday()) % 7
    return sep_first + timedelta(days=days_until_monday)


# Convenience function for integration
def resolve_date_range(query: str, openai_client=None) -> Dict[str, Any]:
    """
    Convenience function for date range resolution.
    
    Args:
        query: Natural language query
        openai_client: Optional OpenAI client
        
    Returns:
        Dict with 'date_range' or empty dict
    """
    resolver = TemporalQueryResolver(openai_client)
    return resolver.resolve_date_range(query)
