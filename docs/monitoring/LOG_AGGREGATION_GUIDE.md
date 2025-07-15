# FreeAgentics Log Aggregation and Analysis Guide

## Overview

This guide documents the log aggregation, parsing, analysis, and streaming capabilities for the FreeAgentics multi-agent system. Our logging infrastructure provides real-time monitoring, structured analysis, and comprehensive search capabilities.

## Table of Contents

1. [Log Architecture](#log-architecture)
2. [Log Sources and Formats](#log-sources-and-formats)
3. [Collection Pipeline](#collection-pipeline)
4. [Log Analysis Patterns](#log-analysis-patterns)
5. [Search and Query Guide](#search-and-query-guide)
6. [Troubleshooting with Logs](#troubleshooting-with-logs)
7. [Log Retention and Compliance](#log-retention-and-compliance)
8. [Best Practices](#best-practices)

## Log Architecture

### Components Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Applications  │────▶│   Log Files     │────▶│    Filebeat     │
│                 │     │                 │     │   (Collector)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Kibana      │◀────│  Elasticsearch  │◀────│    Logstash     │
│  (Visualization)│     │    (Storage)    │     │   (Processing)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │   S3 Archive    │
                        │  (Long-term)    │
                        └─────────────────┘
```

### Data Flow

1. **Generation**: Applications write structured JSON logs
2. **Collection**: Filebeat collects and forwards logs
3. **Processing**: Logstash parses and enriches logs
4. **Storage**: Elasticsearch indexes for search
5. **Visualization**: Kibana provides dashboards
6. **Archive**: S3 for long-term compliance storage

## Log Sources and Formats

### Application Logs

#### Backend API Logs
**Location**: `/var/log/freeagentics/api.log`  
**Format**: JSON structured logging

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "INFO",
  "component": "api",
  "method": "POST",
  "path": "/api/v1/agents",
  "status": 200,
  "duration_ms": 45,
  "request_id": "req-123456",
  "user_id": "user-789",
  "ip": "192.168.1.100",
  "message": "Agent created successfully"
}
```

#### Agent Logs
**Location**: `/var/log/freeagentics/agents/*.log`  
**Format**: JSON with agent context

```json
{
  "timestamp": "2025-01-15T10:30:46.456Z",
  "level": "DEBUG",
  "component": "agent",
  "agent_id": "agent-001",
  "agent_type": "coordinator",
  "coordination_id": "coord-789",
  "belief_state": {
    "free_energy": 2.5,
    "confidence": 0.85
  },
  "memory_mb": 25.3,
  "message": "Coalition formed successfully"
}
```

#### Security Logs
**Location**: `/var/log/freeagentics/security.log`  
**Format**: JSON with security context

```json
{
  "timestamp": "2025-01-15T10:30:47.789Z",
  "level": "WARNING",
  "component": "security",
  "event_type": "authentication",
  "user_id": "user-789",
  "action": "login",
  "outcome": "success",
  "ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "mfa_used": true,
  "message": "User authenticated with MFA"
}
```

#### System Logs
**Location**: `/var/log/freeagentics/system.log`  
**Format**: JSON with system metrics

```json
{
  "timestamp": "2025-01-15T10:30:48.012Z",
  "level": "INFO",
  "component": "system",
  "event": "health_check",
  "memory_usage_mb": 1536,
  "cpu_percent": 45.2,
  "active_agents": 15,
  "goroutines": 234,
  "message": "System health check completed"
}
```

### Log Levels

| Level | Usage | Examples |
|-------|-------|----------|
| DEBUG | Detailed debugging info | Variable values, function entry/exit |
| INFO | General information | Successful operations, state changes |
| WARNING | Warning conditions | High resource usage, retries |
| ERROR | Error conditions | Failed operations, exceptions |
| CRITICAL | Critical failures | System failures, data loss risk |

## Collection Pipeline

### Filebeat Configuration

`/etc/filebeat/filebeat.yml`:

```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/freeagentics/*.log
    json.keys_under_root: true
    json.add_error_key: true
    fields:
      service: freeagentics
      environment: production
    multiline.pattern: '^{'
    multiline.negate: true
    multiline.match: after

  - type: log
    enabled: true
    paths:
      - /var/log/freeagentics/agents/*.log
    json.keys_under_root: true
    fields:
      service: freeagentics-agents
      environment: production

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
  - add_docker_metadata:
      host: "unix:///var/run/docker.sock"
  - drop_event:
      when:
        or:
          - equals:
              level: "DEBUG"
              environment: "production"
  - timestamp:
      field: timestamp
      layouts:
        - '2006-01-02T15:04:05.999Z'
      test:
        - '2025-01-15T10:30:45.123Z'

output.logstash:
  hosts: ["logstash:5044"]
  ssl.certificate_authorities: ["/etc/filebeat/ca.crt"]
  ssl.certificate: "/etc/filebeat/filebeat.crt"
  ssl.key: "/etc/filebeat/filebeat.key"

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0600
```

### Logstash Pipeline

`/etc/logstash/pipeline/freeagentics.conf`:

```ruby
input {
  beats {
    port => 5044
    ssl => true
    ssl_certificate_authorities => ["/etc/logstash/ca.crt"]
    ssl_certificate => "/etc/logstash/logstash.crt"
    ssl_key => "/etc/logstash/logstash.key"
  }
}

filter {
  # Parse JSON logs
  json {
    source => "message"
    target => "parsed"
  }

  # Extract fields
  mutate {
    copy => {
      "[parsed][timestamp]" => "@timestamp"
      "[parsed][level]" => "level"
      "[parsed][component]" => "component"
      "[parsed][message]" => "message"
    }
  }

  # Enrich agent logs
  if [component] == "agent" {
    mutate {
      copy => {
        "[parsed][agent_id]" => "agent_id"
        "[parsed][agent_type]" => "agent_type"
        "[parsed][memory_mb]" => "memory_mb"
      }
    }
    
    # Add memory alert
    if [memory_mb] > 30 {
      mutate {
        add_tag => ["high_memory"]
      }
    }
  }

  # Enrich security logs
  if [component] == "security" {
    geoip {
      source => "[parsed][ip]"
      target => "geoip"
    }
    
    # Flag suspicious activity
    if [parsed][outcome] == "failure" {
      ruby {
        code => '
          event.set("failed_attempts", event.get("failed_attempts").to_i + 1)
          if event.get("failed_attempts") > 5
            event.tag("suspicious_activity")
          end
        '
      }
    }
  }

  # Calculate response time buckets
  if [parsed][duration_ms] {
    if [parsed][duration_ms] <= 100 {
      mutate { add_field => { "performance_bucket" => "fast" } }
    } else if [parsed][duration_ms] <= 500 {
      mutate { add_field => { "performance_bucket" => "normal" } }
    } else {
      mutate { add_field => { "performance_bucket" => "slow" } }
    }
  }

  # Remove processed fields
  mutate {
    remove_field => ["message", "parsed", "host", "agent"]
  }
}

output {
  # Send to Elasticsearch
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    ssl => true
    ssl_certificate_verification => true
    cacert => "/etc/logstash/ca.crt"
    user => "${ELASTIC_USER}"
    password => "${ELASTIC_PASSWORD}"
    index => "freeagentics-%{+YYYY.MM.dd}"
    template_name => "freeagentics"
    template => "/etc/logstash/templates/freeagentics.json"
    template_overwrite => true
  }

  # Send security events to separate index
  if [component] == "security" {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      ssl => true
      user => "${ELASTIC_USER}"
      password => "${ELASTIC_PASSWORD}"
      index => "freeagentics-security-%{+YYYY.MM.dd}"
    }
  }

  # Archive to S3
  s3 {
    region => "us-east-1"
    bucket => "freeagentics-logs-archive"
    prefix => "%{+YYYY/MM/dd}"
    time_file => 300
    codec => "json_lines"
  }

  # Alert on critical errors
  if [level] == "CRITICAL" {
    http {
      url => "${ALERT_WEBHOOK_URL}"
      http_method => "post"
      format => "json"
      mapping => {
        "text" => "Critical error in %{component}: %{message}"
        "level" => "%{level}"
        "timestamp" => "%{@timestamp}"
      }
    }
  }
}
```

### Elasticsearch Index Template

`/etc/logstash/templates/freeagentics.json`:

```json
{
  "index_patterns": ["freeagentics-*"],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index.lifecycle.name": "freeagentics-ilm-policy",
    "index.lifecycle.rollover_alias": "freeagentics"
  },
  "mappings": {
    "properties": {
      "@timestamp": { "type": "date" },
      "level": { "type": "keyword" },
      "component": { "type": "keyword" },
      "message": { "type": "text" },
      "agent_id": { "type": "keyword" },
      "agent_type": { "type": "keyword" },
      "memory_mb": { "type": "float" },
      "duration_ms": { "type": "float" },
      "status": { "type": "integer" },
      "user_id": { "type": "keyword" },
      "request_id": { "type": "keyword" },
      "ip": { "type": "ip" },
      "geoip": {
        "properties": {
          "location": { "type": "geo_point" },
          "country_name": { "type": "keyword" }
        }
      }
    }
  }
}
```

## Log Analysis Patterns

### Common Search Queries

#### Error Analysis
```
# All errors in last hour
level:ERROR AND @timestamp:[now-1h TO now]

# Errors by component
level:ERROR | stats count by component

# Error trends
level:ERROR | timechart count by component
```

#### Performance Analysis
```
# Slow API requests
component:api AND duration_ms:>500

# Average response time by endpoint
component:api | stats avg(duration_ms) by path

# Performance over time
component:api | timechart avg(duration_ms) by path
```

#### Agent Analysis
```
# High memory agents
component:agent AND memory_mb:>30

# Agent errors by type
component:agent AND level:ERROR | stats count by agent_type

# Agent coordination patterns
component:agent AND coordination_id:* | stats count by coordination_id
```

#### Security Analysis
```
# Failed authentications
component:security AND event_type:authentication AND outcome:failure

# Authentication by country
component:security AND event_type:authentication | stats count by geoip.country_name

# Suspicious activity
tags:suspicious_activity
```

### Kibana Dashboards

#### System Overview Dashboard
- Log volume over time
- Error rate by component
- Top error messages
- Component health status

#### Agent Performance Dashboard
- Agent count by type
- Memory usage distribution
- Coordination success rate
- Agent lifecycle events

#### Security Dashboard
- Authentication attempts
- Failed login patterns
- Geographic distribution
- Security alerts

#### API Performance Dashboard
- Request rate
- Response time percentiles
- Error rate by endpoint
- Slow query analysis

## Search and Query Guide

### Elasticsearch Query DSL

#### Basic Queries
```json
// Find all errors
{
  "query": {
    "term": {
      "level": "ERROR"
    }
  }
}

// Find logs in time range
{
  "query": {
    "range": {
      "@timestamp": {
        "gte": "now-1h",
        "lte": "now"
      }
    }
  }
}
```

#### Complex Queries
```json
// Find slow API requests with errors
{
  "query": {
    "bool": {
      "must": [
        { "term": { "component": "api" } },
        { "term": { "level": "ERROR" } },
        { "range": { "duration_ms": { "gte": 500 } } }
      ]
    }
  }
}

// Aggregate by agent memory usage
{
  "aggs": {
    "memory_stats": {
      "stats": {
        "field": "memory_mb"
      }
    },
    "high_memory_agents": {
      "terms": {
        "field": "agent_id",
        "size": 10,
        "order": { "avg_memory": "desc" }
      },
      "aggs": {
        "avg_memory": {
          "avg": { "field": "memory_mb" }
        }
      }
    }
  }
}
```

### Kibana Query Language (KQL)

```
# Simple field search
component:api

# Wildcard search
message:*timeout*

# Range queries
memory_mb:>30

# Boolean operators
component:agent AND (level:ERROR OR level:WARNING)

# Phrase search
message:"connection refused"

# Existence check
agent_id:*
```

## Troubleshooting with Logs

### Investigation Workflows

#### 1. System Outage Investigation
```bash
# Check system status
component:system AND level:(ERROR OR CRITICAL) AND @timestamp:[now-1h TO now]

# Check recent deployments
message:"deployment" OR message:"startup" | sort @timestamp desc

# Check resource issues
message:"memory" OR message:"disk" OR message:"cpu" | stats count by component
```

#### 2. Performance Degradation
```bash
# Identify slow operations
duration_ms:>1000 | stats avg(duration_ms), max(duration_ms) by path

# Check database queries
component:database AND query_time_ms:>100

# Analyze by time window
component:api | timechart avg(duration_ms) span=5m
```

#### 3. Agent Issues
```bash
# Find problematic agents
component:agent AND level:ERROR | stats count by agent_id | sort count desc

# Memory leak detection
component:agent | timechart max(memory_mb) by agent_id

# Coordination failures
message:"coordination failed" | stats count by agent_id, coordination_id
```

#### 4. Security Incidents
```bash
# Failed login attempts
component:security AND outcome:failure | stats count by ip, user_id

# Unusual access patterns
component:security AND NOT geoip.country_name:"United States"

# Permission violations
event_type:access_violation | table timestamp, user_id, resource, action
```

### Log Correlation

#### Request Tracing
```json
// Find all logs for a request
{
  "query": {
    "term": {
      "request_id": "req-123456"
    }
  },
  "sort": [
    { "@timestamp": "asc" }
  ]
}
```

#### Agent Session Tracing
```json
// Find all logs for an agent session
{
  "query": {
    "bool": {
      "must": [
        { "term": { "agent_id": "agent-001" } },
        { "range": { "@timestamp": { "gte": "2025-01-15T10:00:00Z", "lte": "2025-01-15T11:00:00Z" } } }
      ]
    }
  }
}
```

## Log Retention and Compliance

### Retention Policies

| Log Type | Hot Storage | Warm Storage | Cold Storage | Total Retention |
|----------|-------------|--------------|--------------|-----------------|
| Application | 7 days | 30 days | 60 days | 90 days |
| Security | 30 days | 90 days | 275 days | 365 days |
| Agent | 7 days | 30 days | 60 days | 90 days |
| System | 3 days | 14 days | 30 days | 45 days |

### Index Lifecycle Management

```json
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_size": "50GB",
            "max_age": "7d"
          },
          "set_priority": {
            "priority": 100
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": {
            "number_of_shards": 1
          },
          "set_priority": {
            "priority": 50
          }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "freeze": {},
          "set_priority": {
            "priority": 0
          }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

### Compliance Requirements

#### GDPR Compliance
- PII masking in logs
- Right to erasure support
- Data retention limits
- Access audit logs

#### Security Compliance
- Encryption at rest
- Encryption in transit
- Access control
- Audit trail maintenance

### Archive Strategy

```bash
# Daily archive to S3
0 2 * * * /usr/local/bin/archive-logs.sh

# Archive script
#!/bin/bash
DATE=$(date -d "yesterday" +%Y%m%d)
elasticdump \
  --input=http://elasticsearch:9200/freeagentics-${DATE} \
  --output=s3://freeagentics-logs-archive/${DATE}/logs.json.gz \
  --type=data \
  --compress
```

## Best Practices

### Logging Standards

#### 1. Structured Logging
```python
# Good: Structured log with context
logger.info("Agent operation completed", {
    "agent_id": agent_id,
    "operation": "coalition_formation",
    "duration_ms": duration,
    "result": "success"
})

# Bad: Unstructured string concatenation
logger.info(f"Agent {agent_id} completed {operation} in {duration}ms")
```

#### 2. Consistent Fields
- Always include: timestamp, level, component
- Use standard field names across components
- Include correlation IDs (request_id, trace_id)

#### 3. Appropriate Log Levels
```python
# DEBUG: Detailed information
logger.debug("Calculating belief state", {"values": belief_values})

# INFO: Normal operations
logger.info("Agent started", {"agent_id": agent_id})

# WARNING: Warning conditions
logger.warning("Memory usage high", {"memory_mb": 45})

# ERROR: Error conditions
logger.error("Database connection failed", {"error": str(e)})

# CRITICAL: System failures
logger.critical("Coordination service down", {"impact": "all_agents"})
```

### Performance Considerations

#### 1. Asynchronous Logging
```python
import asyncio
from aiologger import Logger

logger = Logger.with_default_handlers()

async def log_async():
    await logger.info("Async log message")
```

#### 2. Sampling
```python
# Sample debug logs in production
if random.random() < 0.01:  # 1% sampling
    logger.debug("Detailed debug info", context)
```

#### 3. Buffering
```yaml
# Filebeat buffer configuration
queue.mem:
  events: 4096
  flush.min_events: 512
  flush.timeout: 5s
```

### Security Best Practices

#### 1. Sensitive Data
```python
# Mask sensitive data
def mask_sensitive(data):
    if "password" in data:
        data["password"] = "***"
    if "token" in data:
        data["token"] = data["token"][:4] + "***"
    return data

logger.info("User login", mask_sensitive(user_data))
```

#### 2. Log Injection Prevention
```python
# Sanitize user input
import re

def sanitize_log_input(text):
    # Remove newlines and control characters
    return re.sub(r'[\n\r\t]', ' ', text)

logger.info("User input", {"input": sanitize_log_input(user_input)})
```

### Monitoring the Logging System

#### Health Checks
```bash
# Check Filebeat
systemctl status filebeat
filebeat test output

# Check Logstash
curl -XGET 'localhost:9600/_node/stats/pipelines?pretty'

# Check Elasticsearch
curl -XGET 'localhost:9200/_cluster/health?pretty'
```

#### Metrics to Monitor
- Log ingestion rate
- Processing lag
- Error rates
- Storage usage

---

**Last Updated**: 2025-01-15  
**Version**: 1.0  
**Contact**: sre@freeagentics.com