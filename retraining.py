import os
import json
import time
import uuid
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

AWS_REGION = "us-east-1"
DYNAMODB_REGION = "ap-south-1"

TABLE_NAME = "campusresolve-retraining-lock"

ROLE_ARN = (
    "arn:aws:iam::134570442470:role/"
    "CampusResolveSageMakerRole"
)

BUCKET = (
    "campusresolve-ai-training-"
    "134570442470-us-east-1"
)

TRAINING_DATA = f"s3://{BUCKET}/training/input/"
SOURCE_ARCHIVE = f"s3://{BUCKET}/training/code/source.tar.gz"
OUTPUT_PATH = f"s3://{BUCKET}/training/output/"

TRAINING_IMAGE = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-cpu-py3"
)

LOCK_ID = "production"

# A STARTING lock without a SageMaker job can be recovered
# after 10 minutes.
STARTING_LOCK_SECONDS = 600

# Wait one hour after a completed or failed job before
# allowing another retraining run.
COOLDOWN_SECONDS = 60

sagemaker = boto3.client(
    "sagemaker",
    region_name=AWS_REGION
)

dynamodb = boto3.resource(
    "dynamodb",
    region_name=DYNAMODB_REGION
)

lock_table = dynamodb.Table(TABLE_NAME)


# --------------------------------------------------
# CHECK EXISTING LOCK
# --------------------------------------------------

def get_lock():
    response = lock_table.get_item(
        Key={"lock_id": LOCK_ID},
        ConsistentRead=True
    )
    return response.get("Item")


def check_existing_job(lock, now):
    """
    Returns True when a current lock must prevent retraining.
    """

    status = lock.get("status")
    job_name = lock.get("training_job_name")

    if status == "COOLDOWN":
        cooldown_until = int(lock.get("cooldown_until", 0))

        if now < cooldown_until:
            logger.info("Retraining cooldown is active.")
            return True

        return False

    if status == "IN_PROGRESS" and job_name:
        try:
            response = sagemaker.describe_training_job(
                TrainingJobName=job_name
            )

            job_status = response["TrainingJobStatus"]

            logger.info(
                "Existing SageMaker job %s has status %s",
                job_name,
                job_status
            )

            if job_status in [
                "InProgress",
                "Starting",
                "Stopping"
            ]:
                return True

            # Job has finished or failed. Start cooldown.
            lock_table.update_item(
                Key={"lock_id": LOCK_ID},
                UpdateExpression=(
                    "SET #s = :cooldown, "
                    "cooldown_until = :until"
                ),
                ExpressionAttributeNames={
                    "#s": "status"
                },
                ExpressionAttributeValues={
                    ":cooldown": "COOLDOWN",
                    ":until": now + COOLDOWN_SECONDS
                }
            )

            logger.info(
                "Job finished with status %s. "
                "Cooldown started.",
                job_status
            )

            return True

        except ClientError:
            logger.exception("Could not inspect existing job.")
            # Fail closed: do not start another job when
            # the existing job's status cannot be verified.
            return True

    if status == "STARTING":
        lease_until = int(lock.get("lease_expires", 0))

        if now < lease_until:
            logger.info("Another invocation is starting training.")
            return True

        # Stale STARTING lock can be reclaimed.
        return False

    return False


# --------------------------------------------------
# ACQUIRE LOCK ATOMICALLY
# --------------------------------------------------

def acquire_lock(owner_token, now):
    try:
        lock_table.put_item(
            Item={
                "lock_id": LOCK_ID,
                "status": "STARTING",
                "owner_token": owner_token,
                "created_at": now,
                "lease_expires": now + STARTING_LOCK_SECONDS
            },
            ConditionExpression=(
                "attribute_not_exists(lock_id) "
                "OR lease_expires < :now "
                "OR cooldown_until < :now"
            ),
            ExpressionAttributeValues={
                ":now": now
            }
        )

        return True

    except ClientError as error:
        if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


# --------------------------------------------------
# START SAGEMAKER TRAINING
# --------------------------------------------------

def start_training_job():
    job_name = (
        "campusresolve-retraining-"
        + str(int(time.time()))
        + "-"
        + uuid.uuid4().hex[:8]
    )

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,

        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
        },

        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": TRAINING_DATA,
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "text/csv"
            }
        ],

        OutputDataConfig={
            "S3OutputPath": OUTPUT_PATH
        },

        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10
        },

        StoppingCondition={
            "MaxRuntimeInSeconds": 3600
        },

        HyperParameters={
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": SOURCE_ARCHIVE
        }
    )

    return job_name


# --------------------------------------------------
# MAIN LAMBDA HANDLER
# --------------------------------------------------

def lambda_handler(event, context):

    logger.info("Received event: %s", json.dumps(event))

    if event.get("drift_detected") is not True:
        return {
            "statusCode": 200,
            "retraining_started": False,
            "message": "Drift not detected. No training started."
        }

    now = int(time.time())
    owner_token = uuid.uuid4().hex

    # Check an existing lock before attempting to acquire one.
    existing_lock = get_lock()

    if existing_lock:
        if check_existing_job(existing_lock, now):
            return {
                "statusCode": 200,
                "retraining_started": False,
                "message": "Existing training job or cooldown prevents retraining.",
                "existing_status": existing_lock.get("status"),
                "training_job_name": existing_lock.get(
                    "training_job_name"
                )
            }

    # Atomic acquisition prevents simultaneous invocations
    # from both starting a training job.
    if not acquire_lock(owner_token, now):
        return {
            "statusCode": 200,
            "retraining_started": False,
            "message": "Another invocation owns the retraining lock."
        }

    try:
        job_name = start_training_job()

        lock_table.update_item(
            Key={"lock_id": LOCK_ID},
            UpdateExpression=(
                "SET #s = :status, "
                "training_job_name = :job, "
                "started_at = :started "
                "REMOVE lease_expires"
            ),
            ConditionExpression="owner_token = :owner",
            ExpressionAttributeNames={
                "#s": "status"
            },
            ExpressionAttributeValues={
                ":status": "IN_PROGRESS",
                ":job": job_name,
                ":started": now,
                ":owner": owner_token
            }
        )

        logger.info("SageMaker training started: %s", job_name)

        return {
            "statusCode": 200,
            "retraining_started": True,
            "training_job_name": job_name,
            "region": AWS_REGION
        }

    except Exception:
        logger.exception("Could not start retraining.")

        # Release only the lock owned by this invocation.
        try:
            lock_table.delete_item(
                Key={"lock_id": LOCK_ID},
                ConditionExpression="owner_token = :owner",
                ExpressionAttributeValues={
                    ":owner": owner_token
                }
            )
        except ClientError:
            logger.exception("Could not release retraining lock.")

        raise
