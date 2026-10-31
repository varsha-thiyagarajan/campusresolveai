
import json
import boto3
import logging
from decimal import Decimal
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("campusresolve-tickets")

lambda_client = boto3.client("lambda")

DRIFT_FUNCTION_NAME = "campusresolve-drift-detection"
MAX_REVIEWS = 25


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
        "body": json.dumps(body, default=decimal_default)
    }


def process_review(review):
    # Validate one review
    if not isinstance(review, dict):
        return {
            "success": False,
            "error": "Each review must be a JSON object"
        }

    ticket_id = review.get("ticket_id")
    final_category = review.get("final_category")
    final_department = review.get("final_department")
    final_priority = review.get("final_priority")
    reviewed_by = review.get("reviewed_by", "admin")

    if not ticket_id:
        return {
            "success": False,
            "error": "ticket_id is required"
        }

    if not final_category:
        return {
            "success": False,
            "ticket_id": ticket_id,
            "error": "final_category is required"
        }

    if not final_department:
        return {
            "success": False,
            "ticket_id": ticket_id,
            "error": "final_department is required"
        }

    if not final_priority:
        return {
            "success": False,
            "ticket_id": ticket_id,
            "error": "final_priority is required"
        }

    reviewed_at = datetime.now(timezone.utc).isoformat()

    # 1. Save human review
    result = table.update_item(
        Key={"ticket_id": ticket_id},
        UpdateExpression="""
            SET
                final_category = :final_category,
                final_department = :final_department,
                final_priority = :final_priority,
                human_reviewed = :human_reviewed,
                review_status = :review_status,
                reviewed_at = :reviewed_at,
                reviewed_by = :reviewed_by
        """,
        ConditionExpression="attribute_exists(ticket_id)",
        ExpressionAttributeValues={
            ":final_category": final_category,
            ":final_department": final_department,
            ":final_priority": final_priority,
            ":human_reviewed": True,
            ":review_status": "COMPLETED",
            ":reviewed_at": reviewed_at,
            ":reviewed_by": reviewed_by
        },
        ReturnValues="ALL_NEW"
    )

    logger.info("Human review saved for ticket %s", ticket_id)

    # 2. Trigger drift detection asynchronously
    drift_triggered = False

    try:
        invoke_result = lambda_client.invoke(
            FunctionName=DRIFT_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps({
                "source": "campusresolve-human-review",
                "ticket_id": ticket_id,
                "reviewed_at": reviewed_at
            }).encode("utf-8")
        )

        drift_triggered = (
            invoke_result.get("StatusCode") == 202
        )

    except Exception:
        logger.exception(
            "Drift invocation failed for ticket %s",
            ticket_id
        )

    # 3. Return individual result
    return {
        "success": True,
        "ticket_id": ticket_id,
        "message": "Human review saved successfully",
        "drift_detection_invocation_accepted": drift_triggered,
        "ticket": result.get("Attributes", {})
    }


def lambda_handler(event, context):
    try:
        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, dict):
            return response(400, {
                "success": False,
                "error": "Request body must be a JSON object"
            })

        # Support both multiple reviews and the old single-review format
        if "reviews" in body:
            reviews = body["reviews"]

            if not isinstance(reviews, list):
                return response(400, {
                    "success": False,
                    "error": "reviews must be a list"
                })

            if len(reviews) == 0:
                return response(400, {
                    "success": False,
                    "error": "At least one review is required"
                })

            if len(reviews) > MAX_REVIEWS:
                return response(400, {
                    "success": False,
                    "error": f"Maximum {MAX_REVIEWS} reviews allowed per request"
                })

        else:
            # Backward compatibility with single-ticket requests
            reviews = [body]

        results = []

        for index, review in enumerate(reviews):
            try:
                result = process_review(review)

            except Exception as exc:
                logger.exception(
                    "Review processing failed at index %s",
                    index
                )

                result = {
                    "success": False,
                    "error": str(exc)
                }

                if isinstance(review, dict):
                    result["ticket_id"] = review.get("ticket_id")

            result["index"] = index
            results.append(result)

        successful = sum(
            1 for item in results if item["success"]
        )

        failed = len(results) - successful

        return response(200, {
            "success": failed == 0,
            "message": "Review processing completed",
            "total_reviews": len(reviews),
            "successful": successful,
            "failed": failed,
            "results": results
        })

    except json.JSONDecodeError:
        return response(400, {
            "success": False,
            "error": "Invalid JSON body"
        })

    except Exception:
        logger.exception("Review processing failed")

        return response(500, {
            "success": False,
            "error": "Unable to process reviews"
        })
