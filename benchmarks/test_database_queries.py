#!/usr/bin/env python3
"""
Database Query Performance Benchmarks
PERF-ENGINEER: Bryan Cantrill + Brendan Gregg Methodology
"""

import pytest
import time
import asyncio
import psutil
import statistics
from typing import List, Dict, Any, Optional
import sqlite3
from dataclasses import dataclass
from contextlib import contextmanager

# Note: This is a placeholder implementation
# In production, replace with actual database connections


@dataclass
class QueryResult:
    """Query execution result."""
    query_type: str
    execution_time: float
    rows_affected: int
    memory_usage: int


class DatabaseBenchmarks:
    """Database query performance benchmarks."""
    
    @pytest.fixture
    def test_db(self):
        """Create test database with sample data."""
        # In-memory SQLite for testing
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                agent_id INTEGER,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY,
                agent_id INTEGER,
                data BLOB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents (id)
            )
        ''')
        
        # Insert test data
        for i in range(1000):
            cursor.execute(
                "INSERT INTO agents (name, state) VALUES (?, ?)",
                (f"agent_{i}", "active" if i % 2 == 0 else "idle")
            )
        
        for i in range(10000):
            cursor.execute(
                "INSERT INTO messages (agent_id, content) VALUES (?, ?)",
                (i % 1000 + 1, f"Message {i}")
            )
        
        conn.commit()
        yield conn
        conn.close()
    
    @contextmanager
    def measure_query(self, query_type: str):
        """Context manager to measure query performance."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        result = QueryResult(
            query_type=query_type,
            execution_time=0,
            rows_affected=0,
            memory_usage=0
        )
        
        yield result
        
        result.execution_time = (time.perf_counter() - start_time) * 1000  # ms
        result.memory_usage = psutil.Process().memory_info().rss - start_memory
    
    @pytest.mark.benchmark(group="database-queries")
    def test_simple_select(self, benchmark, test_db):
        """Benchmark simple SELECT queries."""
        
        def run_query():
            cursor = test_db.cursor()
            cursor.execute("SELECT * FROM agents WHERE state = 'active'")
            results = cursor.fetchall()
            return len(results)
        
        count = benchmark(run_query)
        assert count == 500  # Half should be active
        
        print(f"\nSimple SELECT: retrieved {count} rows")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_join_query(self, benchmark, test_db):
        """Benchmark JOIN queries."""
        
        def run_query():
            cursor = test_db.cursor()
            cursor.execute('''
                SELECT a.name, COUNT(m.id) as message_count
                FROM agents a
                LEFT JOIN messages m ON a.id = m.agent_id
                GROUP BY a.id, a.name
                HAVING COUNT(m.id) > 5
            ''')
            results = cursor.fetchall()
            return len(results)
        
        count = benchmark(run_query)
        assert count > 0
        
        print(f"\nJOIN query: retrieved {count} rows")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_aggregate_query(self, benchmark, test_db):
        """Benchmark aggregate queries."""
        
        def run_query():
            cursor = test_db.cursor()
            cursor.execute('''
                SELECT 
                    agent_id,
                    COUNT(*) as msg_count,
                    MIN(timestamp) as first_msg,
                    MAX(timestamp) as last_msg
                FROM messages
                GROUP BY agent_id
                ORDER BY msg_count DESC
                LIMIT 10
            ''')
            results = cursor.fetchall()
            return len(results)
        
        count = benchmark(run_query)
        assert count == 10
        
        print(f"\nAggregate query: retrieved top {count} agents")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_bulk_insert(self, benchmark, test_db):
        """Benchmark bulk INSERT operations."""
        
        def run_insert():
            cursor = test_db.cursor()
            data = [(f"bulk_agent_{i}", "active") for i in range(100)]
            cursor.executemany(
                "INSERT INTO agents (name, state) VALUES (?, ?)",
                data
            )
            test_db.commit()
            return cursor.rowcount
        
        count = benchmark(run_insert)
        assert count == 100
        
        print(f"\nBulk INSERT: inserted {count} rows")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_update_query(self, benchmark, test_db):
        """Benchmark UPDATE operations."""
        
        def run_update():
            cursor = test_db.cursor()
            cursor.execute(
                "UPDATE agents SET state = 'inactive' WHERE state = 'idle'"
            )
            test_db.commit()
            return cursor.rowcount
        
        count = benchmark(run_update)
        assert count == 500  # Half should be idle
        
        print(f"\nUPDATE: modified {count} rows")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_indexed_vs_non_indexed(self, benchmark, test_db):
        """Compare indexed vs non-indexed queries."""
        
        cursor = test_db.cursor()
        
        # Test without index
        def query_without_index():
            cursor.execute(
                "SELECT * FROM messages WHERE content LIKE 'Message 5%'"
            )
            return cursor.fetchall()
        
        # Create index
        cursor.execute("CREATE INDEX idx_messages_content ON messages(content)")
        test_db.commit()
        
        # Test with index
        def query_with_index():
            cursor.execute(
                "SELECT * FROM messages WHERE content LIKE 'Message 5%'"
            )
            return cursor.fetchall()
        
        # Run benchmark
        without_index_time = benchmark(query_without_index)
        with_index_time = benchmark(query_with_index)
        
        print(f"\nIndexed vs Non-indexed:")
        print(f"  Without index: {len(without_index_time)} rows")
        print(f"  With index: {len(with_index_time)} rows")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_connection_pool_simulation(self, benchmark):
        """Simulate connection pool performance."""
        
        class ConnectionPool:
            def __init__(self, size=10):
                self.connections = []
                for _ in range(size):
                    self.connections.append(sqlite3.connect(':memory:'))
                self.index = 0
            
            def get_connection(self):
                conn = self.connections[self.index % len(self.connections)]
                self.index += 1
                return conn
            
            def close_all(self):
                for conn in self.connections:
                    conn.close()
        
        pool = ConnectionPool()
        
        def run_pooled_queries():
            results = []
            for i in range(100):
                conn = pool.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                results.append(cursor.fetchone())
            return len(results)
        
        count = benchmark(run_pooled_queries)
        assert count == 100
        
        pool.close_all()
        print(f"\nConnection pool: executed {count} queries")
    
    @pytest.mark.benchmark(group="database-queries")
    def test_transaction_performance(self, benchmark, test_db):
        """Benchmark transaction performance."""
        
        def run_with_transaction():
            cursor = test_db.cursor()
            test_db.execute("BEGIN TRANSACTION")
            
            for i in range(100):
                cursor.execute(
                    "INSERT INTO messages (agent_id, content) VALUES (?, ?)",
                    (1, f"Transaction message {i}")
                )
            
            test_db.commit()
            return 100
        
        def run_without_transaction():
            cursor = test_db.cursor()
            
            for i in range(100):
                cursor.execute(
                    "INSERT INTO messages (agent_id, content) VALUES (?, ?)",
                    (1, f"No transaction message {i}")
                )
                test_db.commit()
            
            return 100
        
        # Compare both approaches
        with_tx = benchmark.pedantic(run_with_transaction, rounds=5)
        without_tx = benchmark.pedantic(run_without_transaction, rounds=5)
        
        print(f"\nTransaction performance:")
        print(f"  With transaction: 100 inserts")
        print(f"  Without transaction: 100 inserts")


def run_database_benchmarks():
    """Run all database benchmarks."""
    pytest.main([
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,stddev,median",
        "--benchmark-sort=mean",
        "--benchmark-group-by=group",
        "-s"  # Don't capture output
    ])


if __name__ == "__main__":
    print("="*60)
    print("PERF-ENGINEER: Database Query Performance Benchmarks")
    print("Bryan Cantrill + Brendan Gregg Methodology")
    print("="*60)
    print("\nNOTE: This is a placeholder implementation using SQLite.")
    print("Replace with actual database connections for production.")
    print("="*60)
    
    run_database_benchmarks()