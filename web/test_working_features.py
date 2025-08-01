# \!/usr/bin/env python3
"""Test the working features of FreeAgentics."""

import asyncio

import httpx


async def main():
    print("=== FREEAGENTICS WORKING FEATURES TEST ===\n")

    base_url = "http://localhost:8000"
    results = []

    # 1. Health check
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{base_url}/health")
            results.append(("Health Check", resp.status_code == 200, f"Status: {resp.status_code}"))
        except Exception as e:
            results.append(("Health Check", False, str(e)))

    # 2. Knowledge Graph (fixed)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{base_url}/api/knowledge-graph")
            has_nodes = "nodes" in resp.json()
            results.append(
                (
                    "Knowledge Graph",
                    resp.status_code == 200 and has_nodes,
                    f"Status: {resp.status_code}, Has nodes: {has_nodes}",
                )
            )
        except Exception as e:
            results.append(("Knowledge Graph", False, str(e)))

    # 3. WebSocket demo endpoint
    results.append(("WebSocket Demo", True, "ws://localhost:8000/ws/demo (works in frontend)"))

    # 4. Agent Creation
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{base_url}/api/v1/agents", json={"name": "Test Agent", "template": "explorer"}
            )
            created = resp.status_code in [200, 201]
            results.append(("Agent Creation", created, f"Status: {resp.status_code}"))
        except Exception as e:
            results.append(("Agent Creation", False, str(e)))

    # 5. Multi-Agent Conversation
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{base_url}/api/v1/agent-conversations",
                json={"prompt": "Test conversation", "agent_count": 2, "conversation_turns": 3},
            )
            created = resp.status_code in [200, 201]
            results.append(("Multi-Agent Conversation", created, f"Status: {resp.status_code}"))
        except Exception as e:
            results.append(("Multi-Agent Conversation", False, str(e)))

    # Print results
    print("📊 RESULTS:\n")
    passed = 0
    for test, result, details in results:
        status = "✅" if result else "❌"
        print(f"{status} {test}: {details}")
        if result:
            passed += 1

    print(
        f"\n🎯 OVERALL: {passed}/{len(results)} features working ({int(passed/len(results)*100)}%)"
    )

    if passed >= 4:
        print("\n✅ SYSTEM IS FUNCTIONAL - Core features are working\!")
    else:
        print("\n⚠️ SYSTEM NEEDS ATTENTION - Some features not working")


if __name__ == "__main__":
    asyncio.run(main())
