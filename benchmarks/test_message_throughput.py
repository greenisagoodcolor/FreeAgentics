#!/usr/bin/env python3
"""
Message Throughput Performance Benchmarks
PERF-ENGINEER: Bryan Cantrill + Brendan Gregg Methodology
"""

import asyncio
import json
import pickle
import statistics
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List

import msgpack
import pytest


@dataclass
class Message:
    """Test message structure."""

    id: str
    sender: str
    recipient: str
    content: Any
    timestamp: float
    metadata: Dict[str, Any]


class MessageRouter:
    """Simulated message router for benchmarking."""

    def __init__(self):
        self.routes: Dict[str, List[str]] = {}
        self.handlers: Dict[str, Any] = {}
        self.message_count = 0
        self.total_latency = 0.0

    def register_handler(self, agent_id: str, handler):
        """Register message handler."""
        self.handlers[agent_id] = handler

    def route_message(self, message: Message):
        """Route message to recipient."""
        start_time = time.perf_counter()

        if message.recipient in self.handlers:
            # Simulate processing
            self.handlers[message.recipient](message)

        latency = time.perf_counter() - start_time
        self.message_count += 1
        self.total_latency += latency

        return latency


class MessageThroughputBenchmarks:
    """Comprehensive message throughput benchmarks."""

    @pytest.fixture
    def router(self):
        """Provide message router."""
        return MessageRouter()

    @pytest.fixture
    def sample_messages(self) -> List[Message]:
        """Generate sample messages."""
        messages = []
        for i in range(1000):
            msg = Message(
                id=f"msg-{i}",
                sender=f"agent-{i % 10}",
                recipient=f"agent-{(i + 1) % 10}",
                content={"data": f"test-data-{i}", "value": i},
                timestamp=time.time(),
                metadata={"priority": i % 3, "type": "test"},
            )
            messages.append(msg)
        return messages

    @pytest.mark.benchmark(group="message-throughput", min_rounds=5)
    def test_simple_message_routing(self, benchmark, router, sample_messages):
        """Benchmark simple message routing."""

        # Register handlers
        for i in range(10):
            router.register_handler(f"agent-{i}", lambda msg: None)  # No-op handler

        def route_messages():
            start_time = time.perf_counter()

            for message in sample_messages:
                router.route_message(message)

            duration = time.perf_counter() - start_time
            throughput = len(sample_messages) / duration

            return throughput

        # Run benchmark
        throughput = benchmark(route_messages)

        print("\nSimple Message Routing:")
        print(f"  Throughput: {throughput:.1f} messages/second")
        print(f"  Average latency: {(router.total_latency / router.message_count) * 1000:.3f}ms")

    @pytest.mark.benchmark(group="message-throughput", min_rounds=5)
    def test_concurrent_message_routing(self, benchmark, router, sample_messages):
        """Benchmark concurrent message routing."""

        # Thread-safe message queue
        message_queue = Queue()
        for msg in sample_messages:
            message_queue.put(msg)

        # Register handlers
        for i in range(10):
            router.register_handler(f"agent-{i}", lambda msg: time.sleep(0.0001))  # Simulate work

        def worker():
            while True:
                try:
                    message = message_queue.get(timeout=0.1)
                    router.route_message(message)
                    message_queue.task_done()
                except Exception:
                    break

        def route_messages_concurrent():
            start_time = time.perf_counter()

            # Start workers
            workers = []
            for _ in range(4):
                t = threading.Thread(target=worker)
                t.start()
                workers.append(t)

            # Wait for completion
            message_queue.join()

            # Stop workers
            for t in workers:
                t.join()

            duration = time.perf_counter() - start_time
            throughput = len(sample_messages) / duration

            return throughput

        # Run benchmark
        throughput = benchmark(route_messages_concurrent)

        print("\nConcurrent Message Routing (4 threads):")
        print(f"  Throughput: {throughput:.1f} messages/second")

    @pytest.mark.benchmark(group="message-throughput", min_rounds=3)
    @pytest.mark.asyncio
    async def test_async_message_routing(self, benchmark):
        """Benchmark async message routing."""

        class AsyncMessageRouter:
            def __init__(self):
                self.handlers = {}
                self.processed = 0

            async def route_message(self, message: Message):
                if message.recipient in self.handlers:
                    await self.handlers[message.recipient](message)
                self.processed += 1

        router = AsyncMessageRouter()

        # Register async handlers
        async def async_handler(msg):
            await asyncio.sleep(0.0001)  # Simulate async work

        for i in range(10):
            router.handlers[f"agent-{i}"] = async_handler

        async def route_messages_async():
            start_time = time.perf_counter()

            # Create messages
            messages = []
            for i in range(1000):
                msg = Message(
                    id=f"async-msg-{i}",
                    sender=f"agent-{i % 10}",
                    recipient=f"agent-{(i + 1) % 10}",
                    content={"data": f"test-{i}"},
                    timestamp=time.time(),
                    metadata={},
                )
                messages.append(msg)

            # Route messages concurrently
            tasks = []
            for msg in messages:
                task = asyncio.create_task(router.route_message(msg))
                tasks.append(task)

            await asyncio.gather(*tasks)

            duration = time.perf_counter() - start_time
            throughput = len(messages) / duration

            return throughput

        # Run benchmark
        throughput = await benchmark(route_messages_async)

        print("\nAsync Message Routing:")
        print(f"  Throughput: {throughput:.1f} messages/second")

    @pytest.mark.benchmark(group="message-throughput")
    def test_message_serialization_performance(self, benchmark, sample_messages):
        """Compare message serialization methods."""

        results = {}

        # Test JSON serialization
        json_times = []
        for msg in sample_messages[:100]:
            msg_dict = {
                "id": msg.id,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
            }

            start = time.perf_counter()
            serialized = json.dumps(msg_dict)
            json.loads(serialized)
            duration = time.perf_counter() - start
            json_times.append(duration * 1000)

        results["JSON"] = {"mean": statistics.mean(json_times), "size": len(json.dumps(msg_dict))}

        # Test Pickle serialization
        pickle_times = []
        for msg in sample_messages[:100]:
            start = time.perf_counter()
            serialized = pickle.dumps(msg)
            pickle.loads(serialized)
            duration = time.perf_counter() - start
            pickle_times.append(duration * 1000)

        results["Pickle"] = {"mean": statistics.mean(pickle_times), "size": len(pickle.dumps(msg))}

        # Test MessagePack serialization
        msgpack_times = []
        for msg in sample_messages[:100]:
            msg_dict = {
                "id": msg.id,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata,
            }

            start = time.perf_counter()
            serialized = msgpack.packb(msg_dict)
            msgpack.unpackb(serialized, raw=False)
            duration = time.perf_counter() - start
            msgpack_times.append(duration * 1000)

        results["MessagePack"] = {
            "mean": statistics.mean(msgpack_times),
            "size": len(msgpack.packb(msg_dict)),
        }

        # Print comparison
        print("\nMessage Serialization Performance:")
        for method, stats in results.items():
            print(f"  {method}:")
            print(f"    Mean time: {stats['mean']:.3f}ms")
            print(f"    Size: {stats['size']} bytes")

    @pytest.mark.benchmark(group="message-throughput")
    def test_message_batching_performance(self, benchmark, router, sample_messages):
        """Test performance with message batching."""

        # Register handlers
        for i in range(10):
            router.register_handler(f"agent-{i}", lambda msg: None)

        batch_sizes = [1, 10, 50, 100]
        results = {}

        for batch_size in batch_sizes:
            start_time = time.perf_counter()

            # Process messages in batches
            for i in range(0, len(sample_messages), batch_size):
                batch = sample_messages[i : i + batch_size]

                # Simulate batch processing
                for msg in batch:
                    router.route_message(msg)

            duration = time.perf_counter() - start_time
            throughput = len(sample_messages) / duration

            results[batch_size] = throughput

        # Print batching analysis
        print("\nMessage Batching Performance:")
        baseline = results[1]
        for batch_size, throughput in results.items():
            improvement = (throughput / baseline - 1) * 100
            print(
                f"  Batch size {batch_size}: {throughput:.1f} msg/s "
                f"({improvement:+.1f}% vs single)"
            )

    @pytest.mark.benchmark(group="message-throughput")
    def test_routing_algorithm_performance(self, benchmark):
        """Compare different routing algorithms."""

        # Setup test data
        num_agents = 100
        num_topics = 20
        num_messages = 10000

        # Algorithm 1: Linear search (O(n))
        class LinearRouter:
            def __init__(self):
                self.subscriptions = []  # List of (agent, topics)

            def subscribe(self, agent_id: str, topics: List[str]):
                self.subscriptions.append((agent_id, set(topics)))

            def route(self, topic: str) -> List[str]:
                recipients = []
                for agent_id, topics in self.subscriptions:
                    if topic in topics:
                        recipients.append(agent_id)
                return recipients

        # Algorithm 2: Hash-based (O(1) average)
        class HashRouter:
            def __init__(self):
                self.topic_to_agents = {}  # topic -> set(agents)

            def subscribe(self, agent_id: str, topics: List[str]):
                for topic in topics:
                    if topic not in self.topic_to_agents:
                        self.topic_to_agents[topic] = set()
                    self.topic_to_agents[topic].add(agent_id)

            def route(self, topic: str) -> List[str]:
                return list(self.topic_to_agents.get(topic, []))

        # Setup routers
        linear_router = LinearRouter()
        hash_router = HashRouter()

        # Subscribe agents
        for i in range(num_agents):
            topics = [f"topic-{j}" for j in range(i % num_topics, (i % num_topics) + 5)]
            linear_router.subscribe(f"agent-{i}", topics)
            hash_router.subscribe(f"agent-{i}", topics)

        # Test linear routing
        start = time.perf_counter()
        for i in range(num_messages):
            topic = f"topic-{i % num_topics}"
            linear_router.route(topic)
        linear_time = time.perf_counter() - start

        # Test hash routing
        start = time.perf_counter()
        for i in range(num_messages):
            topic = f"topic-{i % num_topics}"
            hash_router.route(topic)
        hash_time = time.perf_counter() - start

        print(f"\nRouting Algorithm Performance ({num_messages} messages):")
        print(
            f"  Linear search: {linear_time * 1000:.1f}ms "
            f"({num_messages / linear_time:.1f} routes/sec)"
        )
        print(
            f"  Hash lookup: {hash_time * 1000:.1f}ms "
            f"({num_messages / hash_time:.1f} routes/sec)"
        )
        print(f"  Speedup: {linear_time / hash_time:.1f}x")

    @pytest.mark.benchmark(group="message-throughput")
    def test_queue_performance(self, benchmark):
        """Test different queue implementations."""

        import queue
        from collections import deque

        num_messages = 10000

        # Test Python Queue
        q = queue.Queue()
        start = time.perf_counter()
        for i in range(num_messages):
            q.put(i)
        for i in range(num_messages):
            q.get()
        queue_time = time.perf_counter() - start

        # Test deque
        dq = deque()
        start = time.perf_counter()
        for i in range(num_messages):
            dq.append(i)
        for i in range(num_messages):
            dq.popleft()
        deque_time = time.perf_counter() - start

        # Test list (inefficient)
        lst = []
        start = time.perf_counter()
        for i in range(num_messages):
            lst.append(i)
        for i in range(num_messages):
            lst.pop(0)  # O(n) operation!
        list_time = time.perf_counter() - start

        print(f"\nQueue Performance Comparison ({num_messages} operations):")
        print(f"  Queue: {queue_time * 1000:.1f}ms " f"({num_messages / queue_time:.1f} ops/sec)")
        print(f"  Deque: {deque_time * 1000:.1f}ms " f"({num_messages / deque_time:.1f} ops/sec)")
        print(f"  List: {list_time * 1000:.1f}ms " f"({num_messages / list_time:.1f} ops/sec)")


def run_throughput_benchmarks():
    """Run all message throughput benchmarks."""
    pytest.main(
        [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-columns=min,max,mean,stddev,median",
            "--benchmark-sort=mean",
            "--benchmark-group-by=group",
            "--benchmark-warmup=on",
        ]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PERF-ENGINEER: Message Throughput Performance Benchmarks")
    print("Bryan Cantrill + Brendan Gregg Methodology")
    print("=" * 60)

    run_throughput_benchmarks()
