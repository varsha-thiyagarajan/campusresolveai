import boto3
import json
import os
import uuid

transcribe = boto3.client(
    "transcribe",
    region_name="ap-south-1"
)

BUCKET = "campusresolve-ai-data-2026"


def lambda_handler(event, context):

    try:
        # Get audio file from event
        audio_key = event.get("audio_key")

        if not audio_key:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "success": False,
                    "error": "audio_key is required"
                })
            }

        job_name = (
            "campusresolve-transcription-"
            + str(uuid.uuid4())
        )

        media_uri = f"s3://{BUCKET}/{audio_key}"

        response = transcribe.start_transcription_job(
            TranscriptionJobName=job_name,

            Media={
                "MediaFileUri": media_uri
            },

            MediaFormat="wav",

            LanguageCode="en-US",

            OutputBucketName=BUCKET,

            OutputKey=f"transcripts/{job_name}.json"
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "message": "Transcription job started",
                "job_name": job_name,
                "audio": media_uri,
                "status": response[
                    "TranscriptionJob"
                ]["TranscriptionJobStatus"]
            })
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "statusCode": 500,
            "body": json.dumps({
                "success": False,
                "error": str(e)
            })
        }
