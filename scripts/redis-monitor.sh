#!/bin/bash
# Redis Production Monitoring Script for FreeAgentics
# Monitors Redis health, performance, and sends alerts

set -euo pipefail

# Configuration
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-80}"
CONNECTION_THRESHOLD="${CONNECTION_THRESHOLD:-80}"

# Function to send notifications
send_alert() {
    local message="$1"
    local severity="$2"

    echo "[$(date)] ALERT [$severity]: $message"

    if [[ -n "$SLACK_WEBHOOK" ]]; then
        local emoji="⚠️"
        [[ "$severity" == "CRITICAL" ]] && emoji="🚨"
        [[ "$severity" == "INFO" ]] && emoji="ℹ️"

        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$emoji Redis Monitor: $message\"}" \
            "$SLACK_WEBHOOK" || true
    fi
}

# Function to execute Redis commands
redis_cmd() {
    local cmd="$1"
    if [[ -n "$REDIS_PASSWORD" ]]; then
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning "$cmd"
    else
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" "$cmd"
    fi
}

# Function to check Redis connectivity
check_connectivity() {
    echo "Checking Redis connectivity..."
    if redis_cmd "ping" | grep -q "PONG"; then
        echo "✓ Redis is responding"
        return 0
    else
        send_alert "Redis is not responding" "CRITICAL"
        return 1
    fi
}

# Function to check memory usage
check_memory() {
    echo "Checking Redis memory usage..."
    local memory_info
    memory_info=$(redis_cmd "info memory")

    local used_memory_mb
    used_memory_mb=$(echo "$memory_info" | grep "used_memory:" | cut -d: -f2 | tr -d '\r' | awk '{print int($1/1024/1024)}')

    local max_memory_mb
    max_memory_mb=$(echo "$memory_info" | grep "maxmemory:" | cut -d: -f2 | tr -d '\r' | awk '{print int($1/1024/1024)}')

    if [[ "$max_memory_mb" -gt 0 ]]; then
        local memory_usage_pct=$((used_memory_mb * 100 / max_memory_mb))
        echo "Memory usage: ${memory_usage_pct}% (${used_memory_mb}MB / ${max_memory_mb}MB)"

        if [[ "$memory_usage_pct" -gt "$MEMORY_THRESHOLD" ]]; then
            send_alert "High memory usage: ${memory_usage_pct}%" "WARNING"
        fi
    else
        echo "Memory usage: ${used_memory_mb}MB (no limit set)"
    fi
}

# Function to check connections
check_connections() {
    echo "Checking Redis connections..."
    local clients_info
    clients_info=$(redis_cmd "info clients")

    local connected_clients
    connected_clients=$(echo "$clients_info" | grep "connected_clients:" | cut -d: -f2 | tr -d '\r')

    echo "Connected clients: $connected_clients"

    # Check if connection count is unusually high (basic heuristic)
    if [[ "$connected_clients" -gt 100 ]]; then
        send_alert "High connection count: $connected_clients" "WARNING"
    fi
}

# Function to check slow log
check_slow_log() {
    echo "Checking Redis slow log..."
    local slow_log_count
    slow_log_count=$(redis_cmd "slowlog len")

    echo "Slow log entries: $slow_log_count"

    if [[ "$slow_log_count" -gt 10 ]]; then
        send_alert "High slow log count: $slow_log_count entries" "WARNING"
        # Show recent slow queries
        echo "Recent slow queries:"
        redis_cmd "slowlog get 5"
    fi
}

# Function to check persistence
check_persistence() {
    echo "Checking Redis persistence..."
    local persistence_info
    persistence_info=$(redis_cmd "info persistence")

    local aof_enabled
    aof_enabled=$(echo "$persistence_info" | grep "aof_enabled:" | cut -d: -f2 | tr -d '\r')

    local rdb_last_save
    rdb_last_save=$(echo "$persistence_info" | grep "rdb_last_save_time:" | cut -d: -f2 | tr -d '\r')

    echo "AOF enabled: $aof_enabled"
    echo "Last RDB save: $(date -d @$rdb_last_save)"

    # Check if last save was too long ago (more than 1 hour)
    local current_time
    current_time=$(date +%s)
    local time_diff=$((current_time - rdb_last_save))

    if [[ "$time_diff" -gt 3600 ]]; then
        send_alert "Last RDB save was $((time_diff / 3600)) hours ago" "WARNING"
    fi
}

# Function to perform health check
health_check() {
    echo "=== Redis Health Check ==="
    echo "Timestamp: $(date)"
    echo "Host: $REDIS_HOST:$REDIS_PORT"
    echo

    local overall_status="HEALTHY"

    check_connectivity || overall_status="UNHEALTHY"
    echo

    if [[ "$overall_status" == "HEALTHY" ]]; then
        check_memory
        echo
        check_connections
        echo
        check_slow_log
        echo
        check_persistence
        echo
    fi

    echo "Overall status: $overall_status"

    if [[ "$overall_status" == "HEALTHY" ]]; then
        send_alert "Health check completed successfully" "INFO"
    fi
}

# Function to get Redis statistics
get_stats() {
    echo "=== Redis Statistics ==="
    redis_cmd "info all"
}

# Function to reset slow log
reset_slow_log() {
    echo "Resetting Redis slow log..."
    redis_cmd "slowlog reset"
    echo "Slow log reset completed"
}

# Main execution
main() {
    case "${1:-health}" in
        "health")
            health_check
            ;;
        "stats")
            get_stats
            ;;
        "memory")
            check_memory
            ;;
        "connections")
            check_connections
            ;;
        "slowlog")
            check_slow_log
            ;;
        "reset-slowlog")
            reset_slow_log
            ;;
        "persistence")
            check_persistence
            ;;
        *)
            echo "Usage: $0 {health|stats|memory|connections|slowlog|reset-slowlog|persistence}"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
