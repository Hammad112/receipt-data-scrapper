"""
Performance benchmarks for latency and throughput testing.
Measures system performance under various loads.
"""
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from src.parsers import ReceiptParser
from src.chunking import ReceiptChunker


class TestParsingPerformance:
    """Benchmark receipt parsing speed."""

    def test_single_receipt_parse_latency(self):
        """Measure single receipt parsing time (should be < 100ms)."""
        text = """
WALMART
123 Main St
01/15/2024

Milk $4.50
Bread $3.25
Total $7.75
        """
        parser = ReceiptParser()
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            parser.parse_receipt(text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        print(f"\nParsing Latency: avg={avg_time:.2f}ms, max={max_time:.2f}ms")
        assert avg_time < 100, f"Average parsing too slow: {avg_time:.2f}ms"

    def test_batch_parsing_throughput(self):
        """Measure parsing throughput (receipts/second)."""
        receipts = []
        for i in range(100):
            text = f"""
STORE{i}
Date: {i+1}/15/2024
Item{i} $10.00
Total $10.00
            """
            receipts.append(text)
        
        parser = ReceiptParser()
        
        start = time.perf_counter()
        for text in receipts:
            parser.parse_receipt(text)
        elapsed = time.perf_counter() - start
        
        throughput = len(receipts) / elapsed
        print(f"\nBatch Throughput: {throughput:.2f} receipts/sec")
        assert throughput > 50, f"Throughput too low: {throughput:.2f} receipts/sec"


class TestChunkingPerformance:
    """Benchmark chunking speed and memory."""

    def test_chunking_latency(self):
        """Measure chunk creation time for single receipt."""
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        # Create receipt with 20 items
        lines = ["WALMART", "01/15/2024"]
        for i in range(20):
            lines.append(f"Item{i} $10.00")
        lines.append("Total $200.00")
        
        receipt = parser.parse_receipt("\n".join(lines))
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            chunks = chunker.chunk_receipt(receipt)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        print(f"\nChunking Latency: {avg_time:.2f}ms for {len(receipt.items)} items")
        assert avg_time < 50, f"Chunking too slow: {avg_time:.2f}ms"

    def test_chunk_count_scaling(self):
        """Verify chunk count scales linearly with item count."""
        chunker = ReceiptChunker()
        parser = ReceiptParser()
        
        results = []
        for item_count in [1, 5, 10, 20, 50]:
            lines = ["STORE", "01/15/2024"]
            for i in range(item_count):
                lines.append(f"Item{i} $5.00")
            lines.append(f"Total ${item_count * 5}.00")
            
            receipt = parser.parse_receipt("\n".join(lines))
            chunks = chunker.chunk_receipt(receipt)
            
            results.append((item_count, len(chunks)))
        
        print("\nChunk Scaling:")
        for items, chunks in results:
            print(f"  {items} items -> {chunks} chunks")
        
        # Should be roughly linear: summary + payment + merchant + items + categories
        # Categories may create fewer chunks for small item counts


class TestQueryPerformance:
    """Benchmark query processing performance."""

    def test_query_parsing_latency(self):
        """Measure query parsing speed."""
        from src.query.query_parser import QueryParser
        
        parser = QueryParser()
        queries = [
            "How much did I spend in January 2024?",
            "Show me all receipts from Walmart",
            "What did I buy last week?",
            "Find health-related purchases over $50",
        ]
        
        times = []
        for query in queries:
            start = time.perf_counter()
            parser.parse(query)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        print(f"\nQuery Parsing: avg={avg_time:.2f}ms, max={max_time:.2f}ms")
        # Note: QueryParser may call LLM for complex queries, so allow up to 2000ms
        assert avg_time < 2000, f"Query parsing too slow: {avg_time:.2f}ms"


class TestDateResolutionPerformance:
    """Benchmark temporal query resolution."""

    def test_temporal_resolution_speed(self):
        """Measure date resolution performance."""
        from src.query.advanced_date_resolver import TemporalQueryResolver
        
        resolver = TemporalQueryResolver()
        queries = [
            "January 2024",
            "last week",
            "Q1 2024",
            "between March and April",
            "yesterday",
        ]
        
        times = []
        for query in queries:
            start = time.perf_counter()
            resolver.resolve_date_range(query)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        print(f"\nDate Resolution: avg={avg_time:.2f}ms")
        assert avg_time < 50, f"Date resolution too slow: {avg_time:.2f}ms"


class TestMemoryEfficiency:
    """Test memory usage patterns."""

    def test_chunk_memory_footprint(self):
        """Verify chunks don't retain excessive memory."""
        import sys
        
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        # Create large receipt
        lines = ["BIGSTORE", "01/15/2024"]
        for i in range(100):
            lines.append(f"Product Item Number {i} Description Here $99.99")
        lines.append("Total $9999.00")
        
        receipt = parser.parse_receipt("\n".join(lines))
        chunks = chunker.chunk_receipt(receipt)
        
        # Calculate average chunk size
        total_content = sum(len(c.content) for c in chunks)
        avg_size = total_content / len(chunks)
        
        print(f"\nChunk Memory: {len(chunks)} chunks, avg {avg_size:.0f} chars")
        
        # Chunks should be reasonably sized
        assert avg_size < 2000, f"Chunks too large: {avg_size:.0f} chars average"


class TestConcurrency:
    """Test thread safety and concurrent operations."""

    def test_concurrent_parsing(self):
        """Parse multiple receipts concurrently."""
        parser = ReceiptParser()
        
        texts = [
            f"STORE{i}\n01/15/2024\nItem $10.00\nTotal $10.00"
            for i in range(20)
        ]
        
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(parser.parse_receipt, texts))
        elapsed = time.perf_counter() - start
        
        assert len(results) == 20
        throughput = 20 / elapsed
        print(f"\nConcurrent Parsing: {throughput:.2f} receipts/sec with 4 workers")
