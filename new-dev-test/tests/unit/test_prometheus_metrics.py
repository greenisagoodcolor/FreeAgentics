"""Unit tests for Prometheus metrics functionality.

Tests individual metric components without full integration.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class TestMetricDefinitions:
    """Test metric definitions and initialization."""

    def test_required_counters_defined(self):
        """Test that required counters are properly defined."""
        # Create a test registry
        test_registry = CollectorRegistry()

        # Define required counters
        agent_spawn_total = Counter(
            "agent_spawn_total",
            "Total number of agents spawned",
            registry=test_registry,
        )

        kg_node_total = Counter(
            "kg_node_total",
            "Total number of knowledge graph nodes created",
            registry=test_registry,
        )

        # Verify counters are created
        assert agent_spawn_total._name == "agent_spawn_total"
        assert kg_node_total._name == "kg_node_total"

        # Test increment
        agent_spawn_total.inc()
        kg_node_total.inc(5)

        # Verify values (internal testing)
        assert agent_spawn_total._value.get() == 1
        assert kg_node_total._value.get() == 5

    def test_http_metrics_defined(self):
        """Test HTTP request metrics are properly defined."""
        test_registry = CollectorRegistry()

        # HTTP request counter
        http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=test_registry,
        )

        # HTTP request duration histogram
        http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=test_registry,
        )

        # Test usage
        http_requests_total.labels(method="GET", endpoint="/health", status="200").inc()
        http_request_duration_seconds.labels(method="GET", endpoint="/health").observe(0.05)

        assert http_requests_total._name == "http_requests_total"
        assert http_request_duration_seconds._name == "http_request_duration_seconds"

    def test_system_metrics_defined(self):
        """Test system health metrics are properly defined."""
        test_registry = CollectorRegistry()

        # System metrics
        system_cpu_usage_percent = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=test_registry,
        )

        system_memory_usage_bytes = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=test_registry,
        )

        # Test setting values
        system_cpu_usage_percent.set(45.5)
        system_memory_usage_bytes.set(1024 * 1024 * 512)  # 512 MB

        assert system_cpu_usage_percent._value.get() == 45.5
        assert system_memory_usage_bytes._value.get() == 536870912


class TestMetricsFunctionality:
    """Test metrics collection and export functionality."""

    def test_metrics_export_format(self):
        """Test that metrics export in correct Prometheus format."""
        from prometheus_client import generate_latest

        test_registry = CollectorRegistry()

        # Create a simple counter
        test_counter = Counter("test_counter", "A test counter", registry=test_registry)
        test_counter.inc()

        # Generate metrics
        output = generate_latest(test_registry).decode("utf-8")

        # Check format
        assert "# HELP test_counter A test counter" in output
        assert "# TYPE test_counter counter" in output
        assert "test_counter 1.0" in output

    def test_labeled_metrics(self):
        """Test metrics with labels."""
        test_registry = CollectorRegistry()

        # Counter with labels
        request_counter = Counter(
            "api_requests",
            "API requests",
            ["method", "endpoint"],
            registry=test_registry,
        )

        # Increment different label combinations
        request_counter.labels(method="GET", endpoint="/health").inc()
        request_counter.labels(method="POST", endpoint="/agents").inc(2)

        # Generate output
        from prometheus_client import generate_latest

        output = generate_latest(test_registry).decode("utf-8")

        # Check labeled metrics
        assert 'api_requests{endpoint="/health",method="GET"} 1.0' in output
        assert 'api_requests{endpoint="/agents",method="POST"} 2.0' in output

    def test_histogram_metrics(self):
        """Test histogram metrics for timing measurements."""
        test_registry = CollectorRegistry()

        # Create histogram
        duration_histogram = Histogram(
            "request_duration_seconds",
            "Request duration",
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0],
            registry=test_registry,
        )

        # Record some observations
        duration_histogram.observe(0.005)
        duration_histogram.observe(0.05)
        duration_histogram.observe(0.2)

        # Generate output
        from prometheus_client import generate_latest

        output = generate_latest(test_registry).decode("utf-8")

        # Check histogram output
        assert "request_duration_seconds_bucket" in output
        assert "request_duration_seconds_count 3.0" in output
        assert "request_duration_seconds_sum" in output


class TestMetricsIntegrationHelpers:
    """Test helper functions for metrics integration."""

    def test_agent_spawn_counter_helper(self):
        """Test helper function for incrementing agent spawn counter."""
        test_registry = CollectorRegistry()

        # Create counter
        agent_spawn_total = Counter(
            "agent_spawn_total",
            "Total agents spawned",
            ["agent_type"],
            registry=test_registry,
        )

        # Helper function
        def record_agent_spawn(agent_type="default"):
            agent_spawn_total.labels(agent_type=agent_type).inc()

        # Test helper
        record_agent_spawn("active_inference")
        record_agent_spawn("llm")
        record_agent_spawn("llm")

        # Verify counts
        from prometheus_client import generate_latest

        output = generate_latest(test_registry).decode("utf-8")

        assert 'agent_spawn_total{agent_type="active_inference"} 1.0' in output
        assert 'agent_spawn_total{agent_type="llm"} 2.0' in output

    def test_kg_node_counter_helper(self):
        """Test helper function for knowledge graph node counter."""
        test_registry = CollectorRegistry()

        # Create counter
        kg_node_total = Counter(
            "kg_node_total", "Total KG nodes", ["node_type"], registry=test_registry
        )

        # Helper function
        def record_kg_node_creation(node_type="entity", count=1):
            kg_node_total.labels(node_type=node_type).inc(count)

        # Test helper
        record_kg_node_creation("entity", 5)
        record_kg_node_creation("relation", 3)
        record_kg_node_creation("attribute", 2)

        # Verify
        from prometheus_client import generate_latest

        output = generate_latest(test_registry).decode("utf-8")

        assert 'kg_node_total{node_type="entity"} 5.0' in output
        assert 'kg_node_total{node_type="relation"} 3.0' in output
        assert 'kg_node_total{node_type="attribute"} 2.0' in output


class TestMetricsMiddleware:
    """Test metrics collection middleware."""

    def test_http_metrics_middleware_concept(self):
        """Test concept for HTTP metrics middleware."""
        import time

        test_registry = CollectorRegistry()

        # Metrics
        http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=test_registry,
        )

        http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            registry=test_registry,
        )

        # Simulate middleware
        def process_request(method, endpoint):
            start_time = time.time()

            # Simulate request processing
            time.sleep(0.01)
            status = "200"

            # Record metrics
            duration = time.time() - start_time
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

            return status

        # Test
        process_request("GET", "/health")
        process_request("POST", "/api/v1/agents")

        # Verify
        from prometheus_client import generate_latest

        output = generate_latest(test_registry).decode("utf-8")

        assert "http_requests_total" in output
        assert "http_request_duration_seconds" in output
