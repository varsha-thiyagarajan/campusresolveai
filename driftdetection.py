
import json
import boto3
import logging
import os
from collections import Counter
from decimal import Decimal

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = "ap-south-1"

dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION
)

tickets_table = dynamodb.Table(
    "campusresolve-tickets"
)

baseline_table = dynamodb.Table(
    "campusresolve-drift-baseline"
)

# Retraining Lambda is in Mumbai.
lambda_client = boto3.client(
    "lambda",
    region_name=AWS_REGION
)

RETRAINING_FUNCTION_NAME = os.environ.get(
    "RETRAINING_FUNCTION_NAME",
    "campusresolve-start-training"
)

DRIFT_THRESHOLD = Decimal("0.20")


# --------------------------------------------------
# JSON HELPERS
# --------------------------------------------------

def decimal_default(obj):
    if isinstance(obj, Decimal):
        if obj % 1 == 0:
            return int(obj)
        return float(obj)

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(
            body,
            default=decimal_default
        )
    }


def log_result(result, event_type):
    logger.info(
        "%s: %s",
        event_type,
        json.dumps(result, default=decimal_default)
    )


# --------------------------------------------------
# DYNAMODB TICKETS
# --------------------------------------------------

def get_all_tickets():
    """Retrieve all tickets using DynamoDB pagination."""
    items = []
    scan_kwargs = {}

    result = tickets_table.scan(**scan_kwargs)
    items.extend(result.get("Items", []))

    while "LastEvaluatedKey" in result:
        scan_kwargs["ExclusiveStartKey"] = (
            result["LastEvaluatedKey"]
        )

        result = tickets_table.scan(**scan_kwargs)
        items.extend(result.get("Items", []))

    logger.info(
        "Retrieved %d tickets from DynamoDB",
        len(items)
    )

    return items


def get_effective_tickets(items):
    """
    Prefer human-reviewed values when available.
    Do not modify the original DynamoDB records.
    """
    effective_items = []
    reviewed_count = 0

    for item in items:
        effective = dict(item)

        if item.get("human_reviewed") is True:
            reviewed_count += 1

            effective["category"] = (
                item.get("final_category")
                or item.get("category")
            )

            effective["department"] = (
                item.get("final_department")
                or item.get("department")
            )

            effective["priority"] = (
                item.get("final_priority")
                or item.get("priority")
            )

        effective_items.append(effective)

    logger.info(
        "Using human-reviewed values for %d of %d tickets",
        reviewed_count,
        len(items)
    )

    return effective_items


# --------------------------------------------------
# DRIFT CALCULATION
# --------------------------------------------------

def distribution(items, field):
    """Calculate a normalized distribution."""
    values = [
        str(item[field])
        for item in items
        if item.get(field) is not None
    ]

    counts = Counter(values)
    total = len(values)

    if total == 0:
        return {}

    return {
        key: Decimal(str(round(count / total, 4)))
        for key, count in counts.items()
    }


def calculate_drift(baseline, current):
    """
    Calculate total variation distance.
    The result ranges from 0 to 1.
    """
    keys = set(baseline.keys()) | set(current.keys())

    score = Decimal("0")

    for key in keys:
        baseline_value = Decimal(
            str(baseline.get(key, 0))
        )

        current_value = Decimal(
            str(current.get(key, 0))
        )

        score += abs(
            current_value - baseline_value
        )

    return (score / Decimal("2")).quantize(
        Decimal("0.0001")
    )


# --------------------------------------------------
# BASELINE
# --------------------------------------------------

def get_baseline():
    result = baseline_table.get_item(
        Key={"baseline_id": "production"}
    )

    return result.get("Item")


def save_baseline(
    category,
    department,
    priority,
    total_tickets
):
    baseline_item = {
        "baseline_id": "production",
        "category_distribution": category,
        "department_distribution": department,
        "priority_distribution": priority,
        "total_tickets": total_tickets
    }

    baseline_table.put_item(
        Item=baseline_item
    )

    logger.info(
        "Production baseline saved with %d tickets",
        total_tickets
    )

    return baseline_item


# --------------------------------------------------
# AUTOMATIC RETRAINING TRIGGER
# --------------------------------------------------

def trigger_retraining(drift_score):
    """
    Asynchronously invoke the retraining Lambda.
    SageMaker training runs in us-east-1.
    """
    payload = {
        "drift_detected": True,
        "drift_score": float(drift_score)
    }

    logger.warning(
        "Invoking retraining Lambda: %s",
        RETRAINING_FUNCTION_NAME
    )

    result = lambda_client.invoke(
        FunctionName=RETRAINING_FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8")
    )

    status_code = result.get(
        "StatusCode"
    )

    if status_code != 202:
        raise RuntimeError(
            "Unexpected asynchronous invocation status: "
            f"{status_code}"
        )

    logger.info(
        "Retraining invocation accepted. "
        "Function=%s, score=%s",
        RETRAINING_FUNCTION_NAME,
        drift_score
    )

    return {
        "invocation_accepted": True,
        "function_name": RETRAINING_FUNCTION_NAME,
        "invocation_status": status_code
    }


# --------------------------------------------------
# MAIN LAMBDA HANDLER
# --------------------------------------------------

def lambda_handler(event, context):
    logger.info(
        "Starting CampusResolve drift detection"
    )

    logger.info(
        "Request ID: %s",
        context.aws_request_id
    )

    try:
        # 1. Retrieve tickets
        tickets = get_all_tickets()

        if not tickets:
            result = {
                "success": True,
                "message": (
                    "No tickets available for drift detection"
                ),
                "total_tickets": 0,
                "drift_detected": False,
                "retraining_triggered": False
            }

            log_result(result, "NO_TICKETS")
            return response(200, result)

        # 2. Use human-reviewed values
        effective_tickets = get_effective_tickets(
            tickets
        )

        # 3. Current distributions
        current_category = distribution(
            effective_tickets, "category"
        )

        current_department = distribution(
            effective_tickets, "department"
        )

        current_priority = distribution(
            effective_tickets, "priority"
        )

        # 4. Get production baseline
        baseline = get_baseline()

        # 5. Create initial baseline if missing
        if baseline is None:
            baseline = save_baseline(
                current_category,
                current_department,
                current_priority,
                len(tickets)
            )

            result = {
                "success": True,
                "message": (
                    "Initial production baseline created successfully"
                ),
                "total_tickets": len(tickets),
                "baseline_created": True,
                "baseline": {
                    "category": current_category,
                    "department": current_department,
                    "priority": current_priority
                },
                "overall_drift_score": 0.0,
                "drift_threshold": DRIFT_THRESHOLD,
                "drift_detected": False,
                "retraining_triggered": False
            }

            log_result(result, "INITIAL_BASELINE")
            return response(200, result)

        # 6. Read baseline distributions
        baseline_category = baseline.get(
            "category_distribution", {}
        )

        baseline_department = baseline.get(
            "department_distribution", {}
        )

        baseline_priority = baseline.get(
            "priority_distribution", {}
        )

        # 7. Calculate field-level drift
        category_drift = calculate_drift(
            baseline_category,
            current_category
        )

        department_drift = calculate_drift(
            baseline_department,
            current_department
        )

        priority_drift = calculate_drift(
            baseline_priority,
            current_priority
        )

        # 8. Overall drift is the maximum field drift
        overall_drift = max(
            category_drift,
            department_drift,
            priority_drift
        )

        drift_detected = (
            overall_drift >= DRIFT_THRESHOLD
        )

        # 9. Build the result
        human_reviewed_count = sum(
            1
            for ticket in tickets
            if ticket.get("human_reviewed") is True
        )

        result = {
            "success": True,
            "message": "Drift detection completed",
            "total_tickets": len(tickets),
            "human_reviewed_tickets": human_reviewed_count,
            "baseline_tickets": baseline.get(
                "total_tickets"
            ),
            "category": {
                "baseline": baseline_category,
                "current": current_category,
                "drift_score": category_drift
            },
            "department": {
                "baseline": baseline_department,
                "current": current_department,
                "drift_score": department_drift
            },
            "priority": {
                "baseline": baseline_priority,
                "current": current_priority,
                "drift_score": priority_drift
            },
            "overall_drift_score": overall_drift,
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_detected": drift_detected,
            "baseline_type": "persistent_dynamodb_baseline",
            "retraining_triggered": False
        }

        # 10. Log drift results
        log_result(
            result,
            "DRIFT_DETECTION_RESULT"
        )

        # 11. Trigger retraining only when drift is detected
        if drift_detected:
            logger.warning(
                "DRIFT DETECTED: score=%s, threshold=%s",
                overall_drift,
                DRIFT_THRESHOLD
            )

            try:
                retraining_result = trigger_retraining(
                    overall_drift
                )

                result["retraining_triggered"] = True
                result["retraining"] = retraining_result

            except Exception as retraining_error:
                # Preserve the drift result even if the
                # retraining invocation fails.
                logger.exception(
                    "Failed to invoke retraining Lambda"
                )

                result["retraining_triggered"] = False
                result["retraining_error"] = str(
                    retraining_error
                )

        else:
            logger.info(
                "No significant drift: score=%s, threshold=%s",
                overall_drift,
                DRIFT_THRESHOLD
            )

        # 12. Log final result
        log_result(
            result,
            "DRIFT_DETECTION_FINAL"
        )

        return response(200, result)

    except Exception as error:
        logger.exception(
            "Drift detection failed"
        )

        result = {
            "success": False,
            "message": "Drift detection failed",
            "error": str(error)
        }

        log_result(
            result,
            "DRIFT_DETECTION_ERROR"
        )

        return response(500, result)
