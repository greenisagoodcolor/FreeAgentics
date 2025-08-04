"""
Test Result Collection and Reporting
====================================

Minimal implementation for test result aggregation and reporting
following production-grade reliability patterns.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result container."""
    
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float = 0.0
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
        """Create from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass 
class TestSuiteResult:
    """Test suite result aggregation."""
    
    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Total suite execution time."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def summary(self) -> Dict[str, int]:
        """Test result summary."""
        summary = defaultdict(int)
        for result in self.results:
            summary[result.status] += 1
        return dict(summary)
    
    @property
    def passed(self) -> int:
        """Number of passed tests."""
        return self.summary.get("passed", 0)
    
    @property
    def failed(self) -> int:
        """Number of failed tests."""
        return self.summary.get("failed", 0)
    
    @property
    def skipped(self) -> int:
        """Number of skipped tests."""
        return self.summary.get("skipped", 0)
    
    @property 
    def errors(self) -> int:
        """Number of tests with errors."""
        return self.summary.get("error", 0)
    
    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)
    
    def finish(self) -> None:
        """Mark suite as finished."""
        self.end_time = datetime.now()
        logger.info(f"Test suite '{self.suite_name}' finished: {self.summary}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suite_name": self.suite_name,
            "results": [r.to_dict() for r in self.results],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "summary": self.summary,
            "metadata": self.metadata
        }


class TestResultCollector:
    """
    Production-grade test result collection with proper error handling,
    logging, and fallback behavior.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize test result collector.
        
        Args:
            output_dir: Directory for output files (optional)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.current_suite: Optional[TestSuiteResult] = None
        self.suites: List[TestSuiteResult] = []
        self._collection_errors: List[str] = []
        
        logger.info(f"Initialized TestResultCollector with output dir: {self.output_dir}")
    
    def start_suite(self, suite_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start collecting results for a test suite.
        
        Args:
            suite_name: Name of the test suite
            metadata: Optional metadata for the suite
        """
        try:
            if self.current_suite is not None:
                logger.warning(f"Starting new suite '{suite_name}' while '{self.current_suite.suite_name}' is active")
                self.finish_suite()
            
            self.current_suite = TestSuiteResult(
                suite_name=suite_name,
                metadata=metadata or {}
            )
            logger.info(f"Started test suite: {suite_name}")
            
        except Exception as e:
            error_msg = f"Failed to start suite '{suite_name}': {e}"
            logger.error(error_msg)
            self._collection_errors.append(error_msg)
    
    def add_result(self, result: TestResult) -> None:
        """
        Add a test result to the current suite.
        
        Args:
            result: Test result to add
        """
        try:
            if self.current_suite is None:
                logger.warning(f"No active suite for test result: {result.test_name}")
                # Create implicit suite for orphaned results
                self.start_suite("default_suite")
            
            self.current_suite.results.append(result)
            logger.debug(f"Added result for {result.test_name}: {result.status}")
            
        except Exception as e:
            error_msg = f"Failed to add result for '{result.test_name}': {e}"
            logger.error(error_msg)
            self._collection_errors.append(error_msg)
    
    def finish_suite(self) -> Optional[TestSuiteResult]:
        """
        Finish current suite and add to collection.
        
        Returns:
            Finished suite result or None if no active suite
        """
        try:
            if self.current_suite is None:
                logger.warning("No active suite to finish")
                return None
            
            self.current_suite.finish()
            self.suites.append(self.current_suite)
            
            suite_result = self.current_suite
            self.current_suite = None
            
            logger.info(f"Finished suite '{suite_result.suite_name}': {suite_result.summary}")
            return suite_result
            
        except Exception as e:
            error_msg = f"Failed to finish current suite: {e}"
            logger.error(error_msg)
            self._collection_errors.append(error_msg)
            return None
    
    def add_test_passed(self, test_name: str, duration: float = 0.0, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a passed test result."""
        result = TestResult(
            test_name=test_name,
            status="passed",
            duration=duration,
            metadata=metadata or {}
        )
        self.add_result(result)
    
    def add_test_failed(self, test_name: str, error_message: str, 
                       stack_trace: Optional[str] = None, duration: float = 0.0,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a failed test result."""
        result = TestResult(
            test_name=test_name,
            status="failed",
            duration=duration,
            error_message=error_message,
            stack_trace=stack_trace,
            metadata=metadata or {}
        )
        self.add_result(result)
    
    def add_test_skipped(self, test_name: str, reason: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a skipped test result."""
        result = TestResult(
            test_name=test_name,
            status="skipped",
            error_message=reason,
            metadata=metadata or {}
        )
        self.add_result(result)
    
    def add_test_error(self, test_name: str, error_message: str,
                      stack_trace: Optional[str] = None, duration: float = 0.0,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a test with error result."""
        result = TestResult(
            test_name=test_name,
            status="error", 
            duration=duration,
            error_message=error_message,
            stack_trace=stack_trace,
            metadata=metadata or {}
        )
        self.add_result(result) 
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall test execution summary."""
        total_summary = defaultdict(int)
        total_duration = 0.0
        
        for suite in self.suites:
            for status, count in suite.summary.items():
                total_summary[status] += count
            total_duration += suite.duration
        
        return {
            "total_suites": len(self.suites),
            "total_tests": sum(total_summary.values()),
            "passed": total_summary.get("passed", 0),
            "failed": total_summary.get("failed", 0),
            "skipped": total_summary.get("skipped", 0),
            "errors": total_summary.get("error", 0),
            "total_duration": total_duration,
            "collection_errors": len(self._collection_errors)
        }
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save results to JSON file.
        
        Args:
            filename: Output filename (optional, auto-generated if None)
            
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_results_{timestamp}.json"
            
            output_path = self.output_dir / filename
            
            # Ensure current suite is finished
            if self.current_suite is not None:
                self.finish_suite()
            
            data = {
                "summary": self.get_summary(),
                "suites": [suite.to_dict() for suite in self.suites],
                "collection_errors": self._collection_errors,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved test results to: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Failed to save results: {e}"
            logger.error(error_msg)
            self._collection_errors.append(error_msg)
            raise
    
    def print_summary(self) -> None:
        """Print test execution summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        print(f"Total Suites: {summary['total_suites']}")
        print(f"Total Tests:  {summary['total_tests']}")
        print(f"Passed:       {summary['passed']}")
        print(f"Failed:       {summary['failed']}")
        print(f"Skipped:      {summary['skipped']}")
        print(f"Errors:       {summary['errors']}")
        print(f"Duration:     {summary['total_duration']:.2f}s")
        
        if summary['collection_errors'] > 0:
            print(f"Collection Errors: {summary['collection_errors']}")
            
        print("="*60)
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self.current_suite = None
        self.suites.clear()
        self._collection_errors.clear()
        logger.info("Cleared all test results")


# Export main classes
__all__ = ["TestResult", "TestSuiteResult", "TestResultCollector"]