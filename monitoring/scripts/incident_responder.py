#!/usr/bin/env python3
"""
Automated incident response system for FreeAgentics.

Executes predefined playbooks in response to alerts.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from sqlalchemy import create_engine, text

import redis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class IncidentResponder:
    """Automated incident response system."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=6379,
            decode_responses=True,
        )
        self.db_engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER', 'freeagentics')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'freeagentics')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'freeagentics')}"
        )
        self.session = None
        self.active_incidents = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load incident response configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def start(self):
        """Start incident response system."""
        self.session = aiohttp.ClientSession()
        logger.info("Starting FreeAgentics Incident Responder...")

        try:
            # Subscribe to alerts via Redis
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe('freeagentics:alerts')

            # Process alerts
            for message in pubsub.listen():
                if message['type'] == 'message':
                    await self.handle_alert(json.loads(message['data']))

        except KeyboardInterrupt:
            logger.info("Incident responder stopped by user")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        self.redis_client.close()
        self.db_engine.dispose()

    async def handle_alert(self, alert: Dict[str, Any]):
        """Handle incoming alert and execute appropriate playbook."""
        alert_type = alert.get('labels', {}).get('alertname')
        severity = alert.get('labels', {}).get('severity', 'warning')

        logger.info(f"Received alert: {alert_type} (severity: {severity})")

        # Find matching playbook
        playbook = self.find_playbook(alert)
        if not playbook:
            logger.warning(f"No playbook found for alert: {alert_type}")
            return

        # Check if incident already active
        incident_key = f"{alert_type}:{alert.get('labels', {})}"
        if incident_key in self.active_incidents:
            logger.info(f"Incident already active: {incident_key}")
            return

        # Create incident
        incident = {
            'id': f"INC-{int(time.time())}",
            'alert': alert,
            'playbook': playbook,
            'started_at': datetime.utcnow(),
            'status': 'active',
            'steps_completed': [],
        }

        self.active_incidents[incident_key] = incident

        # Execute playbook
        await self.execute_playbook(incident)

    def find_playbook(self, alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find matching playbook for alert."""
        alert_name = alert.get('labels', {}).get('alertname')

        # Match by alert name
        for playbook_name, playbook in self.config['playbooks'].items():
            triggers = playbook.get('triggers', [])
            for trigger in triggers:
                if 'alert' in trigger and trigger['alert'] == alert_name:
                    return playbook

        # Match by metric conditions
        # This would require more complex matching logic

        return None

    async def execute_playbook(self, incident: Dict[str, Any]):
        """Execute incident response playbook."""
        playbook = incident['playbook']
        logger.info(
            f"Executing playbook: {playbook['name']} for incident {incident['id']}"
        )

        try:
            # Execute each step
            for step in playbook['steps']:
                if not await self.should_execute_step(step, incident):
                    logger.info(
                        f"Skipping step: {step['name']} (condition not met)"
                    )
                    continue

                logger.info(f"Executing step: {step['name']}")

                success = await self.execute_step(step, incident)

                if success:
                    incident['steps_completed'].append(step['name'])
                else:
                    logger.error(f"Step failed: {step['name']}")
                    if step.get('required', True):
                        logger.error(
                            f"Required step failed, aborting playbook"
                        )
                        break

                # Wait if specified
                if 'wait' in step:
                    wait_time = self.parse_duration(step['wait'])
                    logger.info(f"Waiting {wait_time}s before next step")
                    await asyncio.sleep(wait_time)

            # Mark incident as resolved
            incident['status'] = 'resolved'
            incident['resolved_at'] = datetime.utcnow()

            logger.info(f"Playbook completed for incident {incident['id']}")

        except Exception as e:
            logger.error(f"Error executing playbook: {e}")
            incident['status'] = 'failed'
            incident['error'] = str(e)

        finally:
            # Remove from active incidents
            incident_key = f"{incident['alert']['labels']['alertname']}:{incident['alert']['labels']}"
            self.active_incidents.pop(incident_key, None)

    async def should_execute_step(
        self, step: Dict[str, Any], incident: Dict[str, Any]
    ) -> bool:
        """Check if step should be executed based on conditions."""
        if 'condition' not in step:
            return True

        condition = step['condition']

        # Simple condition evaluation
        if condition == 'memory_still_high':
            return await self.check_memory_still_high()
        elif condition == 'primary_coordinator_unhealthy':
            return await self.check_coordinator_health()
        elif condition == 'recent_deployment < 30m':
            return await self.check_recent_deployment(30)

        return True

    async def execute_step(
        self, step: Dict[str, Any], incident: Dict[str, Any]
    ) -> bool:
        """Execute a single playbook step."""
        step_type = step['type']

        try:
            if step_type == 'diagnostic':
                return await self.execute_diagnostic_step(step)
            elif step_type == 'remediation':
                return await self.execute_remediation_step(step)
            elif step_type == 'scaling':
                return await self.execute_scaling_step(step)
            elif step_type == 'notification':
                return await self.execute_notification_step(step, incident)
            elif step_type == 'protection':
                return await self.execute_protection_step(step)
            else:
                logger.warning(f"Unknown step type: {step_type}")
                return False

        except Exception as e:
            logger.error(f"Error executing step {step['name']}: {e}")
            return False

    async def execute_diagnostic_step(self, step: Dict[str, Any]) -> bool:
        """Execute diagnostic actions."""
        for action in step['actions']:
            command = action['command']

            if command == 'collect_memory_profile':
                await self.collect_memory_profile()
            elif command == 'list_top_memory_consumers':
                await self.list_top_memory_consumers()
            else:
                logger.warning(f"Unknown diagnostic command: {command}")

        return True

    async def execute_remediation_step(self, step: Dict[str, Any]) -> bool:
        """Execute remediation actions."""
        for action in step['actions']:
            command = action['command']
            params = action.get('params', {})

            if command == 'trigger_gc_all_agents':
                await self.trigger_gc_all_agents(params)
            elif command == 'kill_agents_above_memory_threshold':
                await self.kill_high_memory_agents(params)
            elif command == 'clear_redis_cache':
                await self.clear_redis_cache(params)
            elif command == 'kill_idle_db_connections':
                await self.kill_idle_db_connections(params)
            elif command == 'reset_connection_pool':
                await self.reset_connection_pool(params)
            else:
                logger.warning(f"Unknown remediation command: {command}")

        return True

    async def execute_scaling_step(self, step: Dict[str, Any]) -> bool:
        """Execute scaling actions."""
        for action in step['actions']:
            command = action['command']
            params = action.get('params', {})

            if command == 'scale_out_agent_pool':
                await self.scale_out_agents(params)
            elif command == 'scale_api_servers':
                await self.scale_api_servers(params)
            else:
                logger.warning(f"Unknown scaling command: {command}")

        return True

    async def execute_protection_step(self, step: Dict[str, Any]) -> bool:
        """Execute protection actions."""
        for action in step['actions']:
            command = action['command']
            params = action.get('params', {})

            if command == 'enable_circuit_breaker':
                await self.enable_circuit_breaker(params)
            elif command == 'set_rate_limits':
                await self.set_rate_limits(params)
            elif command == 'cloudflare_under_attack_mode':
                await self.enable_cloudflare_protection(params)
            else:
                logger.warning(f"Unknown protection command: {command}")

        return True

    async def execute_notification_step(
        self, step: Dict[str, Any], incident: Dict[str, Any]
    ) -> bool:
        """Send notifications."""
        for action in step['actions']:
            if 'notify' in action:
                notify_config = action['notify']
                teams = notify_config.get('teams', [])
                severity = notify_config.get('severity', 'info')
                message = notify_config.get('message', 'Incident notification')

                for team in teams:
                    await self.send_notification(
                        team, severity, message, incident
                    )

        return True

    # Implementation of specific actions

    async def collect_memory_profile(self):
        """Collect memory profiling data."""
        logger.info("Collecting memory profile...")

        # Run memory profiling script
        result = subprocess.run(
            [
                'python',
                '/home/green/FreeAgentics/scripts/analyze_agent_memory.py',
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Memory profile collected successfully")
            # Store results
            self.redis_client.set(
                'incident:memory_profile:latest',
                result.stdout,
                ex=3600,  # Expire after 1 hour
            )
        else:
            logger.error(f"Failed to collect memory profile: {result.stderr}")

    async def trigger_gc_all_agents(self, params: Dict[str, Any]):
        """Trigger garbage collection on all agents."""
        logger.info("Triggering GC on all agents...")

        # Send GC command via API
        async with self.session.post(
            'http://localhost:8000/api/v1/agents/gc',
            json={'mode': params.get('mode', 'normal')},
        ) as response:
            if response.status == 200:
                logger.info("GC triggered successfully")
            else:
                logger.error(f"Failed to trigger GC: {response.status}")

    async def kill_high_memory_agents(self, params: Dict[str, Any]):
        """Kill agents exceeding memory threshold."""
        threshold_mb = params.get('threshold_mb', 35)
        preserve_critical = params.get('preserve_critical', True)

        logger.info(f"Killing agents above {threshold_mb}MB...")

        # Get agent memory usage
        async with self.session.get(
            'http://localhost:8000/api/v1/monitoring/metrics/agent_memory'
        ) as response:
            if response.status == 200:
                data = await response.json()

                killed_count = 0
                for agent in data.get('agents', []):
                    if agent['memory_mb'] > threshold_mb:
                        if preserve_critical and agent.get('critical', False):
                            logger.info(
                                f"Preserving critical agent: {agent['id']}"
                            )
                            continue

                        # Kill agent
                        async with self.session.delete(
                            f"http://localhost:8000/api/v1/agents/{agent['id']}"
                        ) as kill_response:
                            if kill_response.status == 200:
                                logger.info(f"Killed agent: {agent['id']}")
                                killed_count += 1

                logger.info(f"Killed {killed_count} agents")

    async def clear_redis_cache(self, params: Dict[str, Any]):
        """Clear Redis cache selectively."""
        preserve = params.get('preserve', [])

        logger.info(f"Clearing Redis cache (preserving: {preserve})...")

        # Get all keys
        keys = self.redis_client.keys('*')

        deleted_count = 0
        for key in keys:
            # Check if key should be preserved
            should_preserve = any(p in key for p in preserve)

            if not should_preserve:
                self.redis_client.delete(key)
                deleted_count += 1

        logger.info(f"Deleted {deleted_count} cache keys")

    async def kill_idle_db_connections(self, params: Dict[str, Any]):
        """Kill idle database connections."""
        idle_time = params.get('idle_time_seconds', 300)
        preserve_count = params.get('preserve_count', 50)

        logger.info(f"Killing connections idle for >{idle_time}s...")

        with self.db_engine.connect() as conn:
            # Get idle connections
            result = conn.execute(
                text(
                    """
                SELECT pid, usename, application_name, state,
                       state_change, query
                FROM pg_stat_activity
                WHERE state = 'idle'
                  AND state_change < now() - interval ':idle_time seconds'
                  AND pid != pg_backend_pid()
                ORDER BY state_change
                LIMIT 1000
            """
                ),
                {'idle_time': idle_time},
            )

            connections = list(result)

            # Keep some connections
            to_kill = connections[preserve_count:]

            killed_count = 0
            for connection in to_kill:
                try:
                    conn.execute(
                        text(f"SELECT pg_terminate_backend({connection.pid})")
                    )
                    killed_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to kill connection {connection.pid}: {e}"
                    )

            logger.info(f"Killed {killed_count} idle connections")

    async def scale_out_agents(self, params: Dict[str, Any]):
        """Scale out agent pool."""
        increment = params.get('increment', 2)
        max_total = params.get('max_total', 50)

        logger.info(f"Scaling out agents by {increment}...")

        # Check current count
        async with self.session.get(
            'http://localhost:8000/api/v1/agents/count'
        ) as response:
            if response.status == 200:
                data = await response.json()
                current_count = data.get('count', 0)

                new_count = min(current_count + increment, max_total)
                to_create = new_count - current_count

                if to_create > 0:
                    # Create new agents
                    for i in range(to_create):
                        async with self.session.post(
                            'http://localhost:8000/api/v1/agents',
                            json={
                                'name': f'scaled-agent-{int(time.time())}-{i}',
                                'type': 'explorer',
                            },
                        ) as create_response:
                            if create_response.status == 201:
                                logger.info(
                                    f"Created new agent {i+1}/{to_create}"
                                )

                    logger.info(f"Scaled out to {new_count} agents")
                else:
                    logger.info(f"Already at max capacity: {max_total}")

    async def enable_circuit_breaker(self, params: Dict[str, Any]):
        """Enable circuit breaker for endpoints."""
        logger.info("Enabling circuit breaker...")

        # Configure circuit breaker via API
        async with self.session.post(
            'http://localhost:8000/api/v1/circuit-breaker/enable', json=params
        ) as response:
            if response.status == 200:
                logger.info("Circuit breaker enabled")
            else:
                logger.error(
                    f"Failed to enable circuit breaker: {response.status}"
                )

    async def set_rate_limits(self, params: Dict[str, Any]):
        """Update rate limiting configuration."""
        logger.info(f"Setting rate limits: {params}")

        # Update rate limits in Redis
        self.redis_client.set(
            'rate_limit:global', params.get('global_rps', 100)
        )
        self.redis_client.set(
            'rate_limit:per_ip', params.get('per_ip_rps', 10)
        )

        logger.info("Rate limits updated")

    async def send_notification(
        self, team: str, severity: str, message: str, incident: Dict[str, Any]
    ):
        """Send notification to team."""
        logger.info(f"Notifying {team}: {message}")

        # Get team notification config
        team_config = self.config.get('notification_channels', {}).get(
            team, {}
        )

        # Send to appropriate channels based on severity
        if severity in ['critical', 'emergency']:
            # Send to all channels
            await self.send_slack_notification(team_config, message, incident)
            await self.send_pagerduty_notification(
                team_config, message, incident
            )
            await self.send_email_notification(team_config, message, incident)
        else:
            # Send only to Slack for warnings
            await self.send_slack_notification(team_config, message, incident)

    async def send_slack_notification(
        self, config: Dict[str, Any], message: str, incident: Dict[str, Any]
    ):
        """Send Slack notification."""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            return

        payload = {
            'text': f"🚨 Incident {incident['id']}: {message}",
            'attachments': [
                {
                    'color': '#ff0000',
                    'fields': [
                        {
                            'title': 'Playbook',
                            'value': incident['playbook']['name'],
                            'short': True,
                        },
                        {
                            'title': 'Started',
                            'value': incident['started_at'].isoformat(),
                            'short': True,
                        },
                    ],
                }
            ],
        }

        try:
            async with self.session.post(
                webhook_url, json=payload
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to send Slack notification: {response.status}"
                    )
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    # Helper methods

    async def check_memory_still_high(self) -> bool:
        """Check if memory usage is still high."""
        async with self.session.get(
            'http://localhost:8000/api/v1/monitoring/metrics/memory'
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('memory_mb', 0) > 1800
        return False

    async def check_coordinator_health(self) -> bool:
        """Check if coordinator is healthy."""
        async with self.session.get(
            'http://localhost:8000/api/v1/coordination/health'
        ) as response:
            return response.status != 200

    async def check_recent_deployment(self, minutes: int) -> bool:
        """Check if there was a recent deployment."""
        # Check deployment timestamp in Redis
        last_deployment = self.redis_client.get('deployment:last_timestamp')
        if last_deployment:
            deployment_time = float(last_deployment)
            return (time.time() - deployment_time) < (minutes * 60)
        return False

    def parse_duration(self, duration: str) -> float:
        """Parse duration string to seconds."""
        if duration.endswith('s'):
            return float(duration[:-1])
        elif duration.endswith('m'):
            return float(duration[:-1]) * 60
        elif duration.endswith('h'):
            return float(duration[:-1]) * 3600
        return float(duration)


async def main():
    """Main entry point."""
    config_path = os.getenv(
        'INCIDENT_RESPONSE_CONFIG',
        '/home/green/FreeAgentics/monitoring/config/incident_response.yaml',
    )

    responder = IncidentResponder(config_path)
    await responder.start()


if __name__ == "__main__":
    asyncio.run(main())
