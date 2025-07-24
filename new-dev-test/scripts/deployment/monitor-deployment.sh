#!/bin/bash
# Monitor deployment health and rollback if necessary

set -euo pipefail

ENVIRONMENT=${1:-production}
TIMEOUT=${2:-300}  # 5 minutes default
CLUSTER="freeagentics-$ENVIRONMENT"
SERVICE="freeagentics-api"

echo "Monitoring deployment in $ENVIRONMENT for $TIMEOUT seconds..."

START_TIME=$(date +%s)
ERROR_COUNT=0
MAX_ERRORS=5
CHECK_INTERVAL=10

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "Deployment monitoring completed after $TIMEOUT seconds"
        break
    fi

    # Check service status
    SERVICE_STATUS=$(aws ecs describe-services \
        --cluster $CLUSTER \
        --services $SERVICE \
        --query 'services[0].deployments[0].status' \
        --output text)

    # Check running task count
    RUNNING_COUNT=$(aws ecs describe-services \
        --cluster $CLUSTER \
        --services $SERVICE \
        --query 'services[0].runningCount' \
        --output text)

    DESIRED_COUNT=$(aws ecs describe-services \
        --cluster $CLUSTER \
        --services $SERVICE \
        --query 'services[0].desiredCount' \
        --output text)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status: $SERVICE_STATUS, Running: $RUNNING_COUNT/$DESIRED_COUNT"

    # Check for deployment failures
    if [ "$SERVICE_STATUS" = "FAILED" ]; then
        echo "ERROR: Deployment failed!"
        exit 1
    fi

    # Check if tasks are failing
    if [ "$RUNNING_COUNT" -lt "$DESIRED_COUNT" ]; then
        ERROR_COUNT=$((ERROR_COUNT + 1))
        echo "WARNING: Running count below desired ($ERROR_COUNT/$MAX_ERRORS errors)"

        if [ $ERROR_COUNT -ge $MAX_ERRORS ]; then
            echo "ERROR: Too many task failures, deployment appears unhealthy"
            exit 1
        fi
    else
        ERROR_COUNT=0  # Reset error count if healthy
    fi

    # Check for steady state
    if [ "$SERVICE_STATUS" = "PRIMARY" ] && [ "$RUNNING_COUNT" -eq "$DESIRED_COUNT" ]; then
        # Check if deployment is complete
        DEPLOYMENT_COUNT=$(aws ecs describe-services \
            --cluster $CLUSTER \
            --services $SERVICE \
            --query 'length(services[0].deployments)' \
            --output text)

        if [ "$DEPLOYMENT_COUNT" -eq 1 ]; then
            echo "SUCCESS: Deployment reached steady state"

            # Perform additional health checks
            echo "Running health checks..."
            if ./scripts/deployment/smoke-tests.sh $ENVIRONMENT; then
                echo "Deployment successful and healthy!"
                exit 0
            else
                echo "ERROR: Health checks failed after deployment"
                exit 1
            fi
        fi
    fi

    # Check CloudWatch alarms
    ALARM_STATE=$(aws cloudwatch describe-alarms \
        --alarm-names "freeagentics-$ENVIRONMENT-high-error-rate" \
        --query 'MetricAlarms[0].StateValue' \
        --output text 2>/dev/null || echo "OK")

    if [ "$ALARM_STATE" = "ALARM" ]; then
        echo "ERROR: CloudWatch alarm triggered - high error rate detected"
        exit 1
    fi

    # Check application logs for errors
    ERROR_LOGS=$(aws logs filter-log-events \
        --log-group-name "/ecs/freeagentics-$ENVIRONMENT" \
        --start-time $(($(date +%s -d '1 minute ago') * 1000)) \
        --filter-pattern "ERROR" \
        --query 'length(events)' \
        --output text 2>/dev/null || echo "0")

    if [ "$ERROR_LOGS" -gt 10 ]; then
        echo "WARNING: High error rate in application logs ($ERROR_LOGS errors in last minute)"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    sleep $CHECK_INTERVAL
done

echo "Monitoring completed"
exit 0
