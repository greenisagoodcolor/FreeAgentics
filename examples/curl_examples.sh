#!/bin/bash
# FreeAgentics API Examples using curl
# Usage: chmod +x curl_examples.sh && ./curl_examples.sh

BASE_URL="http://localhost:8000"
API_BASE="$BASE_URL/api/v1"

echo "FreeAgentics API Examples with curl"
echo "==================================="

# Function to pretty print JSON responses
pretty_print() {
    if command -v jq &> /dev/null; then
        echo "$1" | jq .
    else
        echo "$1"
    fi
}

# Example 1: Health Check
echo -e "\n1. Health Check"
echo "GET $BASE_URL/health"
RESPONSE=$(curl -s "$BASE_URL/health")
pretty_print "$RESPONSE"

# Example 2: Create a Basic Explorer Agent
echo -e "\n2. Create Basic Explorer Agent"
echo "POST $API_BASE/agents"
AGENT_DATA='{
    "name": "curl-explorer-1",
    "agent_type": "explorer",
    "config": {
        "grid_size": 10,
        "use_pymdp": true,
        "exploration_rate": 0.3
    }
}'

RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$AGENT_DATA" \
    "$API_BASE/agents")
pretty_print "$RESPONSE"

# Extract agent ID for subsequent requests
AGENT_ID=$(echo "$RESPONSE" | grep -o '"agent_id":"[^"]*"' | cut -d'"' -f4)
echo "Created agent ID: $AGENT_ID"

# Example 3: Get Agent Information
echo -e "\n3. Get Agent Information"
echo "GET $API_BASE/agents/$AGENT_ID"
RESPONSE=$(curl -s "$API_BASE/agents/$AGENT_ID")
pretty_print "$RESPONSE"

# Example 4: Start the Agent
echo -e "\n4. Start Agent"
echo "POST $API_BASE/agents/$AGENT_ID/start"
RESPONSE=$(curl -s -X POST "$API_BASE/agents/$AGENT_ID/start")
pretty_print "$RESPONSE"

# Example 5: Send Observation to Agent
echo -e "\n5. Send Observation to Agent"
echo "POST $API_BASE/agents/$AGENT_ID/step"
OBSERVATION='{
    "observation": {
        "position": [5, 5],
        "surroundings": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    }
}'

RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$OBSERVATION" \
    "$API_BASE/agents/$AGENT_ID/step")
pretty_print "$RESPONSE"

# Example 6: Get Agent Metrics
echo -e "\n6. Get Agent Metrics"
echo "GET $API_BASE/agents/$AGENT_ID/metrics"
RESPONSE=$(curl -s "$API_BASE/agents/$AGENT_ID/metrics")
pretty_print "$RESPONSE"

# Example 7: List All Agents
echo -e "\n7. List All Agents"
echo "GET $API_BASE/agents"
RESPONSE=$(curl -s "$API_BASE/agents")
pretty_print "$RESPONSE"

# Example 8: Create GMN Agent
echo -e "\n8. Create GMN Agent"
echo "POST $API_BASE/agents/from-gmn"
GMN_DATA='{
    "name": "curl-gmn-agent",
    "gmn_spec": {
        "num_states": [4],
        "num_obs": [4],
        "num_actions": [4],
        "A": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
        "C": [[0.0, 0.0, 0.0, 2.0]]
    },
    "config": {
        "use_pymdp": true,
        "planning_horizon": 3
    }
}'

RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$GMN_DATA" \
    "$API_BASE/agents/from-gmn")
pretty_print "$RESPONSE"

GMN_AGENT_ID=$(echo "$RESPONSE" | grep -o '"agent_id":"[^"]*"' | cut -d'"' -f4)
echo "Created GMN agent ID: $GMN_AGENT_ID"

# Example 9: Create Coalition
echo -e "\n9. Create Coalition"
echo "POST $API_BASE/coalitions"
COALITION_DATA='{
    "name": "curl-test-coalition",
    "objectives": {
        "primary": "explore_efficiently",
        "secondary": "share_information"
    },
    "strategy": "coordinated_search"
}'

RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$COALITION_DATA" \
    "$API_BASE/coalitions")
pretty_print "$RESPONSE"

COALITION_ID=$(echo "$RESPONSE" | grep -o '"coalition_id":"[^"]*"' | cut -d'"' -f4)
echo "Created coalition ID: $COALITION_ID"

# Example 10: Add Agent to Coalition
if [ ! -z "$COALITION_ID" ] && [ ! -z "$AGENT_ID" ]; then
    echo -e "\n10. Add Agent to Coalition"
    echo "POST $API_BASE/coalitions/$COALITION_ID/agents/$AGENT_ID"
    RESPONSE=$(curl -s -X POST "$API_BASE/coalitions/$COALITION_ID/agents/$AGENT_ID")
    pretty_print "$RESPONSE"
fi

# Example 11: Get System Metrics
echo -e "\n11. Get System Metrics - CPU Usage"
echo "GET $API_BASE/metrics/cpu_usage?duration=60"
RESPONSE=$(curl -s "$API_BASE/metrics/cpu_usage?duration=60")
pretty_print "$RESPONSE"

# Example 12: Get Available Metric Types
echo -e "\n12. Get Available Metric Types"
echo "GET $API_BASE/metrics/types"
RESPONSE=$(curl -s "$API_BASE/metrics/types")
pretty_print "$RESPONSE"

# Example 13: Get Performance Counters
echo -e "\n13. Get Performance Counters"
echo "GET $API_BASE/metrics/counters"
RESPONSE=$(curl -s "$API_BASE/metrics/counters")
pretty_print "$RESPONSE"

# Example 14: Update Agent Configuration
echo -e "\n14. Update Agent Configuration"
echo "PUT $API_BASE/agents/$AGENT_ID"
UPDATE_DATA='{
    "config": {
        "exploration_rate": 0.5,
        "planning_horizon": 4
    }
}'

RESPONSE=$(curl -s -X PUT \
    -H "Content-Type: application/json" \
    -d "$UPDATE_DATA" \
    "$API_BASE/agents/$AGENT_ID")
pretty_print "$RESPONSE"

# Example 15: Get GMN Examples
echo -e "\n15. Get GMN Examples"
echo "GET $API_BASE/gmn/examples"
RESPONSE=$(curl -s "$API_BASE/gmn/examples")
pretty_print "$RESPONSE"

# Example 16: Stop Agents
echo -e "\n16. Stop Agents"
echo "POST $API_BASE/agents/$AGENT_ID/stop"
RESPONSE=$(curl -s -X POST "$API_BASE/agents/$AGENT_ID/stop")
pretty_print "$RESPONSE"

if [ ! -z "$GMN_AGENT_ID" ]; then
    echo "POST $API_BASE/agents/$GMN_AGENT_ID/stop"
    RESPONSE=$(curl -s -X POST "$API_BASE/agents/$GMN_AGENT_ID/stop")
    pretty_print "$RESPONSE"
fi

# Example 17: Delete Agents (Cleanup)
echo -e "\n17. Delete Agents (Cleanup)"
echo "DELETE $API_BASE/agents/$AGENT_ID"
RESPONSE=$(curl -s -X DELETE "$API_BASE/agents/$AGENT_ID")
echo "Agent deleted: HTTP status $(curl -s -o /dev/null -w '%{http_code}' -X DELETE "$API_BASE/agents/$AGENT_ID")"

if [ ! -z "$GMN_AGENT_ID" ]; then
    echo "DELETE $API_BASE/agents/$GMN_AGENT_ID"
    RESPONSE=$(curl -s -X DELETE "$API_BASE/agents/$GMN_AGENT_ID")
    echo "GMN Agent deleted: HTTP status $(curl -s -o /dev/null -w '%{http_code}' -X DELETE "$API_BASE/agents/$GMN_AGENT_ID")"
fi

echo -e "\n==================================="
echo "All curl examples completed!"
echo ""
echo "Additional useful commands:"
echo "  # Monitor WebSocket (requires wscat: npm install -g wscat)"
echo "  wscat -c ws://localhost:8000/api/v1/ws/monitor/test-client"
echo ""
echo "  # Get real-time logs"
echo "  tail -f logs/freeagentics.json"
echo ""
echo "  # Check application status"
echo "  curl -s $BASE_URL/health | jq '.overall_status'"
