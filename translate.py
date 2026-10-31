import boto3
import json

translate = boto3.client(
    "translate",
    region_name="ap-south-1"
)

lambda_client = boto3.client(
    "lambda",
    region_name="ap-south-1"
)

TRIAGE_FUNCTION = "campusresolve-triage"


def lambda_handler(event, context):

    try:
        text = event.get("text")
        source_language = event.get("source_language", "ta")

        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "success": False,
                    "error": "text is required"
                })
            }

        # Translate to English
        response = translate.translate_text(
            Text=text,
            SourceLanguageCode=source_language,
            TargetLanguageCode="en"
        )

        translated_text = response["TranslatedText"]

        # Send translated English text
        # to the existing CampusResolve classifier
        triage_payload = {
            "text": translated_text
        }

        triage_response = lambda_client.invoke(
            FunctionName=TRIAGE_FUNCTION,
            InvocationType="RequestResponse",
            Payload=json.dumps(triage_payload).encode("utf-8")
        )

        raw_result = (
            triage_response["Payload"]
            .read()
            .decode("utf-8")
        )

        try:
            triage_result = json.loads(raw_result)
        except json.JSONDecodeError:
            triage_result = raw_result

        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "original_text": text,
                "source_language": source_language,
                "translated_text": translated_text,
                "triage_result": triage_result
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
