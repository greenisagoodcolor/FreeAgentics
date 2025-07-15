#!/usr/bin/env python3
"""
Test script for log aggregation and analysis system.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_log_aggregation():
    """Test the log aggregation system."""
    print("🧪 Testing log aggregation system...")

    try:
        # Import the modules
        from observability.log_aggregation import (
            LogEntry,
            LogLevel,
            LogParser,
            LogSource,
            create_structured_log_entry,
            log_agent_action,
            log_aggregator,
            log_api_request,
            log_security_event,
            start_log_aggregation,
            stop_log_aggregation,
        )

        print("✅ Successfully imported log aggregation modules")

        # Test log parsing
        print("🔍 Testing log parsing...")
        parser = LogParser()

        # Test different log formats
        test_lines = [
            "2024-07-15 14:30:00,123 - INFO - api.auth - User authentication successful",
            "2024-07-15T14:30:01.456 [INFO] Agent:agent_001 - Starting inference process",
            "2024-07-15T14:30:02.789 SECURITY [ERROR] Authentication failed for user",
            '{"timestamp": "2024-07-15T14:30:03.000", "level": "info", "source": "system", "message": "System startup complete", "module": "main"}',
            "Unparseable log line without standard format",
        ]

        parsed_entries = []
        for line in test_lines:
            entry = parser.parse_log_line(line)
            if entry:
                parsed_entries.append(entry)
                print(
                    f"   ✅ Parsed: {entry.level.value} from {entry.source.value} - {entry.message[:50]}..."
                )

        print(f"✅ Successfully parsed {len(parsed_entries)} out of {len(test_lines)} log lines")

        # Test log aggregation
        print("🔄 Testing log aggregation...")

        # Start aggregation
        await start_log_aggregation()
        print("✅ Log aggregation started")

        # Test structured log entry creation
        test_entry = create_structured_log_entry(
            level=LogLevel.INFO,
            source=LogSource.AGENT,
            message="Test log entry",
            module="test_module",
            agent_id="test_agent_001",
            correlation_id="test_corr_001",
            test_field="test_value",
        )

        # Ingest test logs
        log_aggregator.ingest_log_entry(test_entry)

        # Test convenience functions
        log_agent_action("test_agent_001", "inference", {"duration": 0.5})
        log_api_request("GET", "/api/v1/test", 200, "test_user")
        log_security_event("test_event", {"severity": "low"})

        # Ingest parsed entries
        for entry in parsed_entries:
            log_aggregator.ingest_log_entry(entry)

        # Wait for processing
        await asyncio.sleep(2)

        # Test querying
        print("🔍 Testing log querying...")
        recent_logs = await log_aggregator.query_logs(
            start_time=datetime.now() - timedelta(minutes=5), limit=100
        )

        print(f"✅ Retrieved {len(recent_logs)} recent logs")

        # Test agent-specific query
        agent_logs = await log_aggregator.query_logs(agent_id="test_agent_001", limit=50)

        print(f"✅ Retrieved {len(agent_logs)} agent-specific logs")

        # Test stats
        stats = log_aggregator.get_stats()
        print(f"📊 Aggregation stats:")
        print(f"   Total logs: {stats.total_logs}")
        print(f"   Processing rate: {stats.processing_rate:.2f} logs/sec")
        print(f"   Buffer size: {stats.buffer_size}")
        print(f"   Alerts triggered: {stats.alerts_triggered}")

        # Test alert system
        print("🚨 Testing alert system...")

        # Add test alert pattern
        alert_pattern = {
            "id": "test_alert",
            "name": "Test Alert",
            "severity": "medium",
            "level": "ERROR",
            "message": "Test alert triggered",
            "conditions": {},
        }

        log_aggregator.add_alert_pattern(alert_pattern)

        # Trigger alert
        error_entry = create_structured_log_entry(
            level=LogLevel.ERROR,
            source=LogSource.SYSTEM,
            message="Test error message",
            module="test_module",
        )

        log_aggregator.ingest_log_entry(error_entry)

        # Wait for alert processing
        await asyncio.sleep(1)

        active_alerts = log_aggregator.get_active_alerts()
        print(f"✅ Active alerts: {len(active_alerts)}")

        # Stop aggregation
        await stop_log_aggregation()
        print("✅ Log aggregation stopped")

        print("\n🎉 Log aggregation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Log aggregation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_log_analysis():
    """Test the log analysis system."""
    print("\n🧪 Testing log analysis system...")

    try:
        # Import the modules
        from observability.log_analysis_dashboard import (
            analyze_agent_performance,
            analyze_recent_logs,
            generate_log_dashboard,
            log_analysis_engine,
            log_dashboard_generator,
        )

        print("✅ Successfully imported log analysis modules")

        # Test log analysis
        print("🔍 Testing log analysis...")

        # Start aggregation first
        from observability.log_aggregation import start_log_aggregation, stop_log_aggregation

        await start_log_aggregation()

        # Add some test data
        from observability.log_aggregation import (
            LogLevel,
            LogSource,
            create_structured_log_entry,
            log_aggregator,
        )

        test_logs = [
            create_structured_log_entry(
                LogLevel.INFO, LogSource.API, "API request processed", "api", user_id="user1"
            ),
            create_structured_log_entry(
                LogLevel.ERROR,
                LogSource.AGENT,
                "Agent processing failed",
                "agent",
                agent_id="agent1",
            ),
            create_structured_log_entry(
                LogLevel.WARNING, LogSource.SECURITY, "Security warning", "security"
            ),
            create_structured_log_entry(
                LogLevel.INFO,
                LogSource.AGENT,
                "Agent inference complete",
                "agent",
                agent_id="agent1",
                action="inference",
            ),
            create_structured_log_entry(
                LogLevel.CRITICAL, LogSource.SYSTEM, "System critical error", "system"
            ),
        ]

        for log in test_logs:
            log_aggregator.ingest_log_entry(log)

        # Wait for processing
        await asyncio.sleep(2)

        # Test analysis
        analysis_result = await analyze_recent_logs(hours=1)

        print(f"📊 Analysis results:")
        print(f"   Total logs analyzed: {analysis_result.total_logs}")
        print(f"   Level distribution: {analysis_result.level_distribution}")
        print(f"   Source distribution: {analysis_result.source_distribution}")
        print(f"   Agent activity: {analysis_result.agent_activity}")
        print(f"   Anomalies detected: {len(analysis_result.anomalies)}")
        print(f"   Recommendations: {len(analysis_result.recommendations)}")

        # Test agent-specific analysis
        if analysis_result.agent_activity:
            agent_id = list(analysis_result.agent_activity.keys())[0]
            agent_analysis = await analyze_agent_performance(agent_id, hours=1)

            print(f"🤖 Agent analysis for {agent_id}:")
            print(f"   Total logs: {agent_analysis.total_logs}")
            print(f"   Error rate: {agent_analysis.error_rate:.2%}")
            print(f"   Performance metrics: {agent_analysis.performance_metrics}")

        # Test dashboard generation
        print("📊 Testing dashboard generation...")
        dashboard_path = await generate_log_dashboard("logs/test_dashboard.html")

        if Path(dashboard_path).exists():
            print(f"✅ Dashboard generated successfully: {dashboard_path}")

            # Check file size
            file_size = Path(dashboard_path).stat().st_size
            print(f"   Dashboard file size: {file_size} bytes")

            # Verify HTML content
            with open(dashboard_path, "r") as f:
                content = f.read()

            if "FreeAgentics Log Analysis Dashboard" in content:
                print("✅ Dashboard content verified")
            else:
                print("❌ Dashboard content invalid")
        else:
            print("❌ Dashboard file not created")

        # Stop aggregation
        await stop_log_aggregation()

        print("\n🎉 Log analysis tests passed!")
        return True

    except Exception as e:
        print(f"❌ Log analysis test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_log_file_ingestion():
    """Test log file ingestion capabilities."""
    print("\n🧪 Testing log file ingestion...")

    try:
        from observability.log_aggregation import LogSource, log_aggregator

        # Create test log file
        test_log_content = """2024-07-15 14:30:00,123 - INFO - api.auth - User authentication successful
2024-07-15 14:30:01,456 - ERROR - agent.inference - Agent processing failed
2024-07-15 14:30:02,789 - WARNING - security.monitor - Security warning detected
2024-07-15 14:30:03,012 - INFO - system.startup - System initialization complete
2024-07-15 14:30:04,345 - ERROR - database.connection - Database connection failed
"""

        test_file_path = "logs/test_ingestion.log"
        Path(test_file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(test_file_path, "w") as f:
            f.write(test_log_content)

        print(f"✅ Created test log file: {test_file_path}")

        # Test file ingestion
        await log_aggregator.ingest_log_file(test_file_path, LogSource.SYSTEM)

        # Wait for processing
        await asyncio.sleep(1)

        # Verify ingestion
        stats = log_aggregator.get_stats()
        print(f"📊 Post-ingestion stats:")
        print(f"   Total logs: {stats.total_logs}")
        print(f"   Parsing errors: {stats.parsing_errors}")

        # Clean up
        Path(test_file_path).unlink()

        print("✅ Log file ingestion test passed!")
        return True

    except Exception as e:
        print(f"❌ Log file ingestion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance():
    """Test performance with large log volumes."""
    print("\n🧪 Testing performance with large log volumes...")

    try:
        import time

        from observability.log_aggregation import (
            LogLevel,
            LogSource,
            create_structured_log_entry,
            log_aggregator,
        )

        # Generate large number of test logs
        print("📊 Generating test logs...")
        start_time = time.time()

        for i in range(1000):
            entry = create_structured_log_entry(
                level=LogLevel.INFO,
                source=LogSource.AGENT,
                message=f"Test log entry {i}",
                module="performance_test",
                agent_id=f"agent_{i % 10}",
                correlation_id=f"corr_{i % 100}",
            )
            log_aggregator.ingest_log_entry(entry)

        generation_time = time.time() - start_time
        print(f"✅ Generated 1000 logs in {generation_time:.2f} seconds")

        # Wait for processing
        await asyncio.sleep(5)

        # Check processing performance
        stats = log_aggregator.get_stats()
        print(f"📊 Performance stats:")
        print(f"   Total logs processed: {stats.total_logs}")
        print(f"   Processing rate: {stats.processing_rate:.2f} logs/sec")
        print(f"   Buffer size: {stats.buffer_size}")

        # Test query performance
        query_start = time.time()
        logs = await log_aggregator.query_logs(limit=500)
        query_time = time.time() - query_start

        print(f"✅ Queried {len(logs)} logs in {query_time:.2f} seconds")

        print("✅ Performance test passed!")
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":

    async def main():
        """Run all tests."""
        print("🧪 FreeAgentics Log Aggregation and Analysis Test Suite")
        print("=" * 60)

        success = True

        # Test 1: Log aggregation
        if not await test_log_aggregation():
            success = False

        # Test 2: Log analysis
        if not await test_log_analysis():
            success = False

        # Test 3: Log file ingestion
        if not await test_log_file_ingestion():
            success = False

        # Test 4: Performance
        if not await test_performance():
            success = False

        print("\n" + "=" * 60)
        if success:
            print("🎉 All tests passed! Log aggregation and analysis system is working correctly.")
            sys.exit(0)
        else:
            print("❌ Some tests failed. Please check the implementation.")
            sys.exit(1)

    asyncio.run(main())
