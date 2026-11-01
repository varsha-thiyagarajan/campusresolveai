
import boto3
import json
import os
import tarfile
import io

REGION = "us-east-1"
BUCKET = "campusresolve-ai-training-134570442470-us-east-1"

MIN_RECORDS = 200
MIN_ACCURACY = 0.75
MIN_MACRO_F1 = 0.75

sagemaker = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

REQUIRED_FILES = [
    "category_model.joblib",
    "department_model.joblib",
    "priority_model.joblib",
    "evaluation_metrics.json",
    "training_metadata.json"
]

TASKS = ["category", "department", "priority"]


def reject(reasons, details):
    return {
        "validation_passed": False,
        "deployment_allowed": False,
        "status": "REJECTED",
        "reasons": reasons,
        "details": details
    }


def lambda_handler(event, context):
    job_name = event.get("training_job_name")
    package_arn = event.get("model_package_arn")

    if not job_name or not package_arn:
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "training_job_name and model_package_arn are required"
            })
        }

    try:
        job = sagemaker.describe_training_job(
            TrainingJobName=job_name
        )

        if job["TrainingJobStatus"] != "Completed":
            return {
                "statusCode": 200,
                "body": json.dumps(reject(
                    ["Training job is not completed"],
                    {"training_status": job["TrainingJobStatus"]}
                ))
            }

        artifact_uri = job["ModelArtifacts"]["S3ModelArtifacts"]

        if not artifact_uri.startswith(f"s3://{BUCKET}/"):
            return {
                "statusCode": 200,
                "body": json.dumps(reject(
                    ["Artifact is outside the approved bucket"], {}
                ))
            }

        key = artifact_uri.replace(f"s3://{BUCKET}/", "", 1)
        response = s3.get_object(Bucket=BUCKET, Key=key)

        with tarfile.open(
            fileobj=io.BytesIO(response["Body"].read()),
            mode="r:gz"
        ) as archive:

            members = {
                os.path.basename(member.name): member
                for member in archive.getmembers()
                if member.isfile()
            }

            missing = [
                name for name in REQUIRED_FILES
                if name not in members
            ]

            if missing:
                return {
                    "statusCode": 200,
                    "body": json.dumps(reject(
                        ["Required model artifacts are missing"],
                        {"missing_files": missing}
                    ))
                }

            metadata = json.load(
                archive.extractfile(members["training_metadata.json"])
            )

            metrics = json.load(
                archive.extractfile(members["evaluation_metrics.json"])
            )

        reasons = []
        details = {}

        # 1. Training record count
        records = metadata.get("training_records_unique", 0)
        details["training_records_unique"] = records

        if records < MIN_RECORDS:
            reasons.append(
                f"Training records below minimum: {records} < {MIN_RECORDS}"
            )

        # 2. Numerical quality checks
        details["quality_metrics"] = {}

        for task in TASKS:
            task_metrics = metrics.get(task, {})
            accuracy = task_metrics.get("accuracy")
            macro_f1 = task_metrics.get("macro_f1")

            details["quality_metrics"][task] = {
                "accuracy": accuracy,
                "macro_f1": macro_f1
            }

            if accuracy is None or accuracy < MIN_ACCURACY:
                reasons.append(f"{task} accuracy below {MIN_ACCURACY}")

            if macro_f1 is None or macro_f1 < MIN_MACRO_F1:
                reasons.append(f"{task} macro F1 below {MIN_MACRO_F1}")

        # 3. Check candidate metadata
        metadata_approval = metadata.get("deployment_approval")
        details["metadata_approval"] = metadata_approval

        if metadata_approval != "REQUIRES_FURTHER_REVIEW":
            reasons.append(
                "Training metadata is not marked for further review"
            )

        # 4. Check Model Registry package
        package = sagemaker.describe_model_package(
            ModelPackageName=package_arn
        )

        registry_approval = package["ModelApprovalStatus"]
        package_status = package["ModelPackageStatus"]

        details["registry_approval"] = registry_approval
        details["package_status"] = package_status
        details["model_package_arn"] = package_arn

        if package_status != "Completed":
            reasons.append("Model package is not Completed")

        if registry_approval != "PendingManualApproval":
            reasons.append(
                "Model package is not awaiting manual approval"
            )

        # 5. Final decision
        if reasons:
            return {
                "statusCode": 200,
                "body": json.dumps(reject(reasons, details))
            }

        return {
            "statusCode": 200,
            "body": json.dumps({
                "validation_passed": True,
                "deployment_allowed": False,
                "status": "VALIDATED_PENDING_APPROVAL",
                "training_job_name": job_name,
                "artifact_uri": artifact_uri,
                "details": details
            })
        }

    except Exception as e:
        print("VALIDATION ERROR:", str(e))

        return {
            "statusCode": 500,
            "body": json.dumps({
                "validation_passed": False,
                "deployment_allowed": False,
                "status": "ERROR",
                "error": str(e)
            })
        }
