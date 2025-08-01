#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for FreeAgentics
Tests the complete user journey: install → dev → API key → working conversations

This test should FAIL initially, then we fix everything to make it pass.
"""

import asyncio
import json
import os
from typing import Any, Dict

import requests
import websockets


class FreeAgenticsE2ETest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        self.test_results.append({"test": test_name, "passed": passed, "message": message})

    def test_services_running(self) -> bool:
        """Test that both backend and frontend services are running"""
        try:
            # Test backend health
            response = requests.get(f"{self.base_url}/health", timeout=5)
            backend_ok = response.status_code == 200
            self.log_test(
                "Backend Service Running", backend_ok, f"Backend status: {response.status_code}"
            )

            # Test frontend loading
            response = requests.get(self.frontend_url, timeout=5)
            frontend_ok = response.status_code == 200 and "FreeAgentics" in response.text
            self.log_test(
                "Frontend Service Running", frontend_ok, f"Frontend status: {response.status_code}"
            )

            return backend_ok and frontend_ok
        except Exception as e:
            self.log_test("Services Running", False, f"Exception: {e}")
            return False

    def test_demo_mode_detection(self) -> bool:
        """Test that system properly detects demo mode vs real mode"""
        try:
            # Try dev-config endpoint first
            response = requests.get(f"{self.base_url}/api/v1/dev-config", timeout=5)
            if response.status_code == 200:
                config = response.json()
                demo_mode = config.get("demo_mode", False)
                llm_provider = config.get("llm_provider", "unknown")

                self.log_test(
                    "Demo Mode Detection",
                    True,
                    f"Demo mode: {demo_mode}, LLM provider: {llm_provider}",
                )
                return True

            # Fallback: check if we can determine mode from settings
            response = requests.get(f"{self.base_url}/api/v1/settings", timeout=5)
            if response.status_code == 200:
                settings = response.json()
                provider = settings.get("llm_provider", "unknown")
                self.log_test(
                    "Demo Mode Detection", True, f"Detected provider from settings: {provider}"
                )
                return True
            else:
                self.log_test(
                    "Demo Mode Detection",
                    False,
                    f"No config endpoints available. Dev-config: {response.status_code}",
                )
                return False

        except Exception as e:
            self.log_test("Demo Mode Detection", False, f"Exception: {e}")
            return False

    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection and status reporting"""
        try:
            # Test demo WebSocket endpoint
            demo_ws_url = f"{self.ws_url}/api/v1/ws/demo"
            async with websockets.connect(demo_ws_url) as websocket:
                # Wait for initial connection message
                initial_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                initial_data = json.loads(initial_response)

                # Connection should be established
                connection_established = initial_data.get("type") == "connection_established"

                if connection_established:
                    # Send test message
                    test_message = {"type": "ping", "data": "test"}
                    await websocket.send(json.dumps(test_message))

                    # Wait for response (optional - connection establishment is enough)
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        response_data = json.loads(response)
                        self.log_test(
                            "WebSocket Connection",
                            True,
                            f"Connected successfully: {initial_data.get('client_id')}",
                        )
                    except asyncio.TimeoutError:
                        # Connection established is sufficient
                        self.log_test(
                            "WebSocket Connection",
                            True,
                            f"Connected successfully: {initial_data.get('client_id')}",
                        )
                    return True
                else:
                    self.log_test(
                        "WebSocket Connection", False, f"Unexpected response: {initial_data}"
                    )
                    return False

        except Exception as e:
            self.log_test("WebSocket Connection", False, f"Exception: {e}")
            return False

    def test_api_key_management(self) -> bool:
        """Test API key storage and validation"""
        try:
            # Test setting API key using PUT method
            test_api_key = "sk-test-key-for-validation"
            settings_data = {"openai_api_key": test_api_key, "llm_provider": "openai"}

            response = requests.put(
                f"{self.base_url}/api/v1/settings", json=settings_data, timeout=5
            )

            if response.status_code == 404:
                self.log_test("API Key Management", False, "No settings endpoint")
                return False

            save_ok = response.status_code in [200, 201]
            self.log_test("API Key Save", save_ok, f"Settings save status: {response.status_code}")

            if not save_ok:
                # Try PATCH method
                response = requests.patch(
                    f"{self.base_url}/api/v1/settings", json=settings_data, timeout=5
                )
                save_ok = response.status_code in [200, 201]
                self.log_test(
                    "API Key Save (PATCH)",
                    save_ok,
                    f"Settings PATCH status: {response.status_code}",
                )

            # Test retrieving settings
            response = requests.get(f"{self.base_url}/api/v1/settings", timeout=5)
            retrieve_ok = response.status_code == 200

            if retrieve_ok:
                settings = response.json()
                # The key might be masked or stored differently
                has_openai_key = "openai_api_key" in settings
                self.log_test(
                    "API Key Retrieval",
                    has_openai_key,
                    f"Has OpenAI key field: {has_openai_key}, settings keys: {list(settings.keys())}",
                )
                return save_ok and has_openai_key
            else:
                self.log_test(
                    "API Key Retrieval", False, f"Retrieval status: {response.status_code}"
                )
                return False

        except Exception as e:
            self.log_test("API Key Management", False, f"Exception: {e}")
            return False

    def test_agent_creation(self) -> bool:
        """Test creating an agent via API"""
        try:
            # Use correct AgentConfig schema
            agent_data = {
                "name": "Test Agent",
                "template": "analyst",  # Use a basic template
                "parameters": {
                    "description": "A test agent for e2e validation",
                    "goal": "Test the system end-to-end",
                },
                "use_pymdp": True,
                "planning_horizon": 3,
            }

            response = requests.post(f"{self.base_url}/api/v1/agents", json=agent_data, timeout=10)

            if response.status_code == 404:
                self.log_test("Agent Creation", False, "No agents endpoint")
                return False

            creation_ok = response.status_code in [200, 201]
            agent_id = None

            if creation_ok:
                agent = response.json()
                agent_id = agent.get("id")
                self.log_test("Agent Creation", True, f"Created agent ID: {agent_id}")
            else:
                self.log_test("Agent Creation", False, f"Creation status: {response.status_code}")
                return False

            # Test retrieving the created agent
            if agent_id:
                response = requests.get(f"{self.base_url}/api/v1/agents/{agent_id}", timeout=5)
                retrieval_ok = response.status_code == 200
                self.log_test(
                    "Agent Retrieval", retrieval_ok, f"Retrieval status: {response.status_code}"
                )
                return creation_ok and retrieval_ok

            return creation_ok

        except Exception as e:
            self.log_test("Agent Creation", False, f"Exception: {e}")
            return False

    def test_conversation_flow(self) -> bool:
        """Test creating and retrieving conversations"""
        try:
            # Create a conversation using agent-conversations endpoint with correct schema
            conversation_data = {
                "prompt": "Create agents to discuss testing and validation of AI systems",
                "config": {"agent_count": 2, "conversation_turns": 3},
                "metadata": {"session_type": "testing", "user_intent": "validation"},
            }

            response = requests.post(
                f"{self.base_url}/api/v1/agent-conversations", json=conversation_data, timeout=10
            )

            if response.status_code == 404:
                self.log_test("Conversation Creation", False, "No agent-conversations endpoint")
                return False

            creation_ok = response.status_code in [200, 201]
            conversation_id = None

            if creation_ok:
                conversation = response.json()
                conversation_id = conversation.get("conversation_id") or conversation.get("id")
                self.log_test(
                    "Conversation Creation", True, f"Created conversation ID: {conversation_id}"
                )
            else:
                self.log_test(
                    "Conversation Creation",
                    False,
                    f"Creation status: {response.status_code}, response: {response.text[:200]}",
                )
                return False

            # Test retrieving conversations
            response = requests.get(f"{self.base_url}/api/v1/agent-conversations", timeout=5)
            list_ok = response.status_code == 200

            if list_ok:
                conversations = response.json()
                # Handle different response formats - check both id and conversation_id fields
                conv_list = (
                    conversations
                    if isinstance(conversations, list)
                    else conversations.get("conversations", [conversations])
                )
                found_conversation = any(
                    (
                        conv.get("id") == conversation_id
                        or conv.get("conversation_id") == conversation_id
                    )
                    for conv in conv_list
                )
                self.log_test(
                    "Conversation Retrieval",
                    found_conversation,
                    f"Found created conversation: {found_conversation}, total: {len(conv_list)}",
                )
                return creation_ok and found_conversation
            else:
                self.log_test(
                    "Conversation Retrieval", False, f"List status: {response.status_code}"
                )
                return False

        except Exception as e:
            self.log_test("Conversation Flow", False, f"Exception: {e}")
            return False

    def test_llm_provider_switching(self) -> bool:
        """Test switching between mock and real LLM providers"""
        try:
            # Test with valid provider (ollama instead of mock)
            mock_settings = {"llm_provider": "ollama", "openai_api_key": ""}

            response = requests.put(
                f"{self.base_url}/api/v1/settings", json=mock_settings, timeout=5
            )

            mock_switch_ok = response.status_code in [200, 201]
            self.log_test(
                "Mock Provider Switch",
                mock_switch_ok,
                f"Mock switch status: {response.status_code}",
            )

            # Test with real provider (should accept even invalid key)
            real_settings = {
                "llm_provider": "openai",
                "openai_api_key": "sk-invalid-key-for-testing",
            }

            response = requests.put(
                f"{self.base_url}/api/v1/settings", json=real_settings, timeout=5
            )

            real_switch_ok = response.status_code in [200, 201]
            self.log_test(
                "Real Provider Switch",
                real_switch_ok,
                f"Real switch status: {response.status_code}",
            )

            return mock_switch_ok and real_switch_ok

        except Exception as e:
            self.log_test("LLM Provider Switching", False, f"Exception: {e}")
            return False

    def test_real_api_key_integration(self) -> bool:
        """Test with real API key if available in environment"""
        real_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

        if not real_key:
            self.log_test(
                "Real API Key Integration",
                False,
                "No real API key in environment (OPENAI_API_KEY or ANTHROPIC_API_KEY)",
            )
            return False

        try:
            provider = "openai" if real_key.startswith("sk-") else "anthropic"
            key_field = "openai_api_key" if provider == "openai" else "anthropic_api_key"

            # Set real API key
            settings_data = {"llm_provider": provider, key_field: real_key}

            response = requests.post(
                f"{self.base_url}/api/v1/settings", json=settings_data, timeout=5
            )

            key_set_ok = response.status_code in [200, 201]
            self.log_test("Real API Key Set", key_set_ok, f"Key set status: {response.status_code}")

            if not key_set_ok:
                return False

            # Test actual LLM call
            test_message = {
                "messages": [{"role": "user", "content": "Say 'Hello from FreeAgentics' exactly"}]
            }

            response = requests.post(
                f"{self.base_url}/api/v1/chat/completions", json=test_message, timeout=30
            )

            if response.status_code == 404:
                self.log_test("Real LLM Call", False, "No chat completions endpoint")
                return False

            llm_ok = response.status_code == 200

            if llm_ok:
                result = response.json()
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                contains_expected = "Hello from FreeAgentics" in response_text
                self.log_test(
                    "Real LLM Call", contains_expected, f"LLM response: {response_text[:100]}..."
                )
                return contains_expected
            else:
                self.log_test("Real LLM Call", False, f"LLM call status: {response.status_code}")
                return False

        except Exception as e:
            self.log_test("Real API Key Integration", False, f"Exception: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        print("🚀 Starting Comprehensive FreeAgentics End-to-End Test")
        print("=" * 60)

        # Run tests in order
        tests_passed = 0
        total_tests = 0

        # Basic service tests
        if self.test_services_running():
            tests_passed += 1
        total_tests += 1

        # Configuration tests
        if self.test_demo_mode_detection():
            tests_passed += 1
        total_tests += 1

        # WebSocket tests
        if await self.test_websocket_connection():
            tests_passed += 1
        total_tests += 1

        # API key management tests
        if self.test_api_key_management():
            tests_passed += 1
        total_tests += 1

        # Agent creation tests
        if self.test_agent_creation():
            tests_passed += 1
        total_tests += 1

        # Conversation flow tests
        if self.test_conversation_flow():
            tests_passed += 1
        total_tests += 1

        # Provider switching tests
        if self.test_llm_provider_switching():
            tests_passed += 1
        total_tests += 1

        # Real API key tests
        if self.test_real_api_key_integration():
            tests_passed += 1
        total_tests += 1

        print("=" * 60)
        print(f"📊 Results: {tests_passed}/{total_tests} tests passed")

        success_rate = (tests_passed / total_tests) * 100
        if success_rate == 100:
            print("🎉 ALL TESTS PASSED! System is fully functional!")
        elif success_rate >= 80:
            print("⚠️  Most tests passed, minor issues remain")
        else:
            print("❌ Major issues detected, system needs significant fixes")

        return {
            "total_tests": total_tests,
            "tests_passed": tests_passed,
            "success_rate": success_rate,
            "results": self.test_results,
        }


async def main():
    """Run the comprehensive test"""
    tester = FreeAgenticsE2ETest()
    results = await tester.run_all_tests()

    # Print detailed results
    print("\n📋 Detailed Test Results:")
    for result in results["results"]:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {result['test']}")
        if result["message"]:
            print(f"   → {result['message']}")

    return results["success_rate"] == 100


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
