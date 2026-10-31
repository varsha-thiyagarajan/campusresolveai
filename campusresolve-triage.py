import json
import math
import os
import re
import time
import uuid
from decimal import Decimal

import boto3
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BUCKET = os.environ.get(
    "BUCKET",
    "campusresolve-ai-data-2026"
)

MODEL_KEY = os.environ.get(
    "MODEL_KEY",
    "models/v1/model_weights.json"
)

TABLE_NAME = os.environ.get(
    "TABLE_NAME"
)

MARGIN_MIN = float(
    os.environ.get(
        "MARGIN_MIN",
        "0.2"
    )
)

MODEL_VERSION = "sagemaker-v1"

_MODELS = None


# ============================================================
# SLA CONFIGURATION
# ============================================================

SLA_HOURS = {
    "Urgent": 1,
    "High": 2,
    "Medium": 8,
    "Low": 24
}


# ============================================================
# PRIORITY RULES
# ============================================================

URGENT_PATTERNS = [
    r"\bfire\b",
    r"\bsmoke\b",
    r"short[- ]?circuit",
    r"electric(al)? shock",
    r"\bsparks?\b",
    r"exposed wires?",
    r"gas leak",
    r"\bragging\b",
    r"harass",
    r"assault",
    r"molest",
    r"\bthreat",
    r"suicid",
    r"self[- ]?harm",
    r"ambulance",
    r"medical emergency",
    r"\bunconscious\b",
    r"stuck in (the )?(lift|elevator)",
    r"\bsnake\b",
    r"collaps",
    r"stampede"
]

HIGH_PATTERNS = [
    r"deadline (is )?(today|tomorrow)",
    r"exam (is )?(today|tomorrow)",
    r"(money|payment) (was )?deducted",
    r"broken (glass|handrail)",
    r"no (drinking )?water"
]

ORDER = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Urgent": 3
}


# ============================================================
# METADATA
# ============================================================

META_KEYS = (
    "source_type",
    "source_key",
    "ocr_confidence",
    "pii_types",
    "language",
    "sentiment"
)


# ============================================================
# LOAD SAGEMAKER-TRAINED MODELS
# ============================================================


def load_models():
    """Load the three trained sklearn Pipelines from the Lambda package."""
    global _MODELS

    if _MODELS is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_files = {
            "category": "category_model.joblib",
            "department": "department_model.joblib",
            "priority": "priority_model.joblib",
        }
        loaded = {}
        for task, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Required trained model is missing: {filename}"
                )
            loaded[task] = joblib.load(path)
        _MODELS = loaded

    return _MODELS


# ============================================================
# TEXT FEATURES
# ============================================================

def word_terms(text, ngram):

    toks = re.findall(
        r"(?u)\b\w\w+\b",
        text.lower()
    )

    terms = list(toks)

    for n in range(
        2,
        ngram[1] + 1
    ):

        for i in range(
            len(toks) - n + 1
        ):

            terms.append(
                " ".join(
                    toks[i:i + n]
                )
            )

    return terms


def char_wb_terms(text, ngram):

    text = re.sub(
        r"\s\s+",
        " ",
        text.lower()
    )

    out = []

    for w in text.split():

        w = " " + w + " "

        for n in range(
            ngram[0],
            ngram[1] + 1
        ):

            offset = 0

            while offset + n <= len(w):

                out.append(
                    w[offset:offset + n]
                )

                offset += 1

    return out


# ============================================================
# TF-IDF
# ============================================================

def tfidf(terms, idf):

    counts = {}

    for term in terms:

        if term in idf:

            counts[term] = (
                counts.get(term, 0) + 1
            )

    vec = {}

    for term, count in counts.items():

        try:

            # IMPORTANT:
            # Convert model value to float.
            # This fixes:
            # "can't multiply sequence by non-int of type 'float'"

            idf_value = float(
                idf[term]
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        vec[term] = (
            (1.0 + math.log(count))
            * idf_value
        )

    norm = math.sqrt(
        sum(
            value * value
            for value in vec.values()
        )
    )

    if norm > 0:

        return {
            term: value / norm
            for term, value in vec.items()
        }

    return {}


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(task, text):
    """Predict a label and a decision-score margin with a trained Pipeline."""
    models = load_models()
    if task not in models:
        raise ValueError(f"Unknown prediction task: {task}")

    model = models[task]
    label = str(model.predict([text])[0])
    classes = [str(value) for value in model.classes_]

    # LinearSVC uses one-vs-rest decision scores for this trained pipeline.
    scores = model.decision_function([text])
    if getattr(scores, "ndim", 1) == 2:
        scores = scores[0]
    else:
        scores = [float(scores[0]), -float(scores[0])]

    scores = [float(value) for value in scores]
    if len(scores) < 2:
        margin = 0.0
    else:
        ordered = sorted(scores, reverse=True)
        margin = round(ordered[0] - ordered[1], 3)

    return label, margin


# ============================================================
# EXPLAINABLE PREDICTION
# ============================================================

def explain_prediction(task, text, top_n=5):
    """Return the prediction and strongest feature contributions when available."""
    model = load_models()[task]
    prediction = str(model.predict([text])[0])
    classes = [str(value) for value in model.classes_]
    scores = model.decision_function([text])
    if getattr(scores, "ndim", 1) == 2:
        scores = scores[0]
    else:
        scores = [float(scores[0]), -float(scores[0])]
    scores = [float(value) for value in scores]
    ordered_scores = sorted(scores, reverse=True)
    margin = round(
        ordered_scores[0] - ordered_scores[1], 3
    ) if len(ordered_scores) > 1 else 0.0

    evidence = []
    try:
        features = model.named_steps["featureunion"]
        classifier = model.named_steps["linearsvc"]
        transformed = features.transform([text])
        feature_names = features.get_feature_names_out()
        class_index = classes.index(prediction)
        if len(classes) == 2 and classifier.coef_.shape[0] == 1:
            # Binary LinearSVC has one coefficient row for classes_[1].
            coef_index = 0
            sign = 1.0 if class_index == 1 else -1.0
        else:
            coef_index = class_index
            sign = 1.0
        row = transformed.tocsr()
        start, end = row.indptr[0], row.indptr[1]
        indices = row.indices[start:end]
        values = row.data[start:end]
        coefficients = classifier.coef_[coef_index]
        contributions = []
        for index, value in zip(indices, values):
            contribution = float(value) * float(coefficients[index]) * sign
            if contribution > 0:
                contributions.append((contribution, str(feature_names[index])))
        contributions.sort(reverse=True)
        for contribution, feature in contributions[:top_n]:
            evidence.append({
                "feature": feature,
                "type": "model_feature",
                "contribution": round(contribution, 4),
            })
    except (AttributeError, KeyError, IndexError, ValueError, TypeError):
        # Keep triage operational even if a future model has a different structure.
        evidence = []

    return {
        "prediction": prediction,
        "evidence": evidence,
        "margin": margin,
    }


# ============================================================
# SAFETY RULES
# ============================================================

def apply_rules(
    text,
    priority
):

    low = text.lower()

    # Urgent rules
    for pattern in URGENT_PATTERNS:

        if re.search(
            pattern,
            low
        ):

            return (
                "Urgent",
                pattern
            )

    # High rules
    for pattern in HIGH_PATTERNS:

        if (
            re.search(
                pattern,
                low
            )
            and ORDER[priority]
            < ORDER["High"]
        ):

            return (
                "High",
                pattern
            )

    return priority, None


# ============================================================
# TRIAGE
# ============================================================

def triage(
    text,
    meta=None
):

    meta = meta or {}

    # --------------------------------------------------------
    # ML predictions
    # --------------------------------------------------------

    category, category_margin = predict(
        "category",
        text
    )

    priority_ml, priority_margin = predict(
        "priority",
        text
    )

    department, department_margin = predict(
        "department",
        text
    )

    # --------------------------------------------------------
    # Apply safety rules
    # --------------------------------------------------------

    priority, rule_fired = apply_rules(
        text,
        priority_ml
    )

    # --------------------------------------------------------
    # Human review
    # --------------------------------------------------------

    word_count = len(
        re.findall(
            r"\w+",
            text
        )
    )

    too_short = (
        word_count < 3
    )

    needs_human_review = (
        too_short
        or min(
            category_margin,
            priority_margin,
            department_margin
        ) < MARGIN_MIN
    )

    # --------------------------------------------------------
    # OCR confidence
    # --------------------------------------------------------

    if (
        meta.get(
            "source_type",
            "text"
        ) != "text"
        and float(
            meta.get(
                "ocr_confidence",
                100
            )
        ) < 80
    ):

        needs_human_review = True

    # --------------------------------------------------------
    # Explainable AI
    # --------------------------------------------------------

    category_explanation = (
        explain_prediction(
            "category",
            text
        )
    )

    priority_explanation = (
        explain_prediction(
            "priority",
            text
        )
    )

    department_explanation = (
        explain_prediction(
            "department",
            text
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "category": category,

        "priority": priority,

        "priority_ml": priority_ml,

        "rule_fired": rule_fired,

        "department": department,

        "margins": {

            "category": category_margin,

            "priority": priority_margin,

            "department": department_margin
        },

        "needs_human_review":
            needs_human_review,

        "model_version":
            MODEL_VERSION,

        "source_type":
            meta.get(
                "source_type",
                "text"
            ),

        "explanation": {

            "category": {

                "prediction":
                    category_explanation[
                        "prediction"
                    ],

                "evidence":
                    category_explanation[
                        "evidence"
                    ],

                "margin":
                    category_margin
            },

            "priority": {

                "ml_prediction":
                    priority_ml,

                "final_prediction":
                    priority,

                "evidence":
                    priority_explanation[
                        "evidence"
                    ],

                "margin":
                    priority_margin
            },

            "department": {

                "prediction":
                    department_explanation[
                        "prediction"
                    ],

                "evidence":
                    department_explanation[
                        "evidence"
                    ],

                "margin":
                    department_margin
            }
        }
    }


# ============================================================
# DYNAMODB CONVERSION
# ============================================================

def to_dynamo(value):

    if isinstance(
        value,
        float
    ):

        return Decimal(
            str(value)
        )

    if isinstance(
        value,
        dict
    ):

        return {
            key: to_dynamo(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        list
    ):

        return [
            to_dynamo(val)
            for val in value
        ]

    return value


# ============================================================
# SAVE TICKET
# ============================================================

def save_ticket(
    text,
    result,
    meta=None
):

    meta = {
        key: value
        for key, value in (
            meta or {}
        ).items()

        if (
            key in META_KEYS
            and value not in (
                None,
                "",
                []
            )
        )
    }

    # --------------------------------------------------------
    # Creation time
    # --------------------------------------------------------

    created_at = int(
        time.time()
    )

    # --------------------------------------------------------
    # SLA
    # --------------------------------------------------------

    priority = result.get(
        "priority",
        "Medium"
    )

    sla_hours = SLA_HOURS.get(
        priority,
        24
    )

    sla_deadline = (
        created_at
        + (
            sla_hours
            * 60
            * 60
        )
    )

    # --------------------------------------------------------
    # Ticket
    # --------------------------------------------------------

    ticket = {

        "ticket_id": str(
            uuid.uuid4()
        ),

        "created_at":
            created_at,

        "complaint":
            text,

        "status": (
            "NEEDS_REVIEW"
            if result[
                "needs_human_review"
            ]
            else "OPEN"
        ),

        # Existing AI result
        **result,

        # ----------------------------------------------------
        # SLA fields
        # ----------------------------------------------------

        "sla_hours":
            sla_hours,

        "sla_deadline":
            sla_deadline,

        "escalated":
            False,

        "escalation_reason":
            None,

        # Metadata
        **meta
    }

    # --------------------------------------------------------
    # Convert values for DynamoDB
    # --------------------------------------------------------

    ticket = to_dynamo(
        ticket
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dynamodb = boto3.resource(
        "dynamodb"
    )

    table = dynamodb.Table(
        TABLE_NAME
    )

    table.put_item(
        Item=ticket
    )

    return ticket[
        "ticket_id"
    ]


# ============================================================
# RESPONSE
# ============================================================

def respond(
    status_code,
    body
):

    return {

        "statusCode":
            status_code,

        "headers": {

            "Content-Type":
                "application/json",

            "Access-Control-Allow-Origin":
                "*"
        },

        "body":
            json.dumps(body)
    }



def lambda_handler(event, context):

    payload = event

    # Handle API Gateway requests
    if isinstance(event.get("body"), str):
        try:
            payload = json.loads(event["body"])
        except json.JSONDecodeError:
            return respond(400, {
                "error": "Request body must contain valid JSON"
            })

    # --------------------------------------------------------
    # SUPPORT SINGLE AND MULTIPLE COMPLAINTS
    # --------------------------------------------------------

    if "complaints" in payload:
        complaints = payload["complaints"]

        if not isinstance(complaints, list) or not complaints:
            return respond(400, {
                "error": "'complaints' must be a non-empty list"
            })

        # Keep initial batches manageable
        if len(complaints) > 25:
            return respond(400, {
                "error": "Maximum 25 complaints allowed per request"
            })

    elif "text" in payload:
        complaints = [payload["text"]]

    else:
        return respond(400, {
            "error": "Provide 'text' or 'complaints'"
        })

    # --------------------------------------------------------
    # VALIDATE AND PROCESS EACH COMPLAINT
    # --------------------------------------------------------

    results = []

    for index, complaint in enumerate(complaints):

        try:
            if not isinstance(complaint, str):
                raise ValueError("Complaint must be a string")

            complaint = complaint.strip()

            if not complaint:
                raise ValueError("Complaint cannot be empty")

            if len(complaint) > 2000:
                raise ValueError(
                    "Complaint exceeds 2000 characters"
                )

            # Metadata can be shared across the batch
            meta = {
                key: payload[key]
                for key in META_KEYS
                if key in payload
            }

            # Run existing prediction pipeline
            result = triage(complaint, meta)

            # Save individual ticket
            if TABLE_NAME:
                result["ticket_id"] = save_ticket(
                    complaint,
                    result,
                    meta
                )

            results.append({
                "index": index,
                "success": True,
                "complaint": complaint,
                **result
            })

        except Exception as e:

            print(
                f"ERROR processing complaint {index}: {str(e)}"
            )

            results.append({
                "index": index,
                "success": False,
                "complaint": (
                    complaint
                    if isinstance(complaint, str)
                    else None
                ),
                "error": str(e)
            })

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    successful = sum(
        1 for item in results if item["success"]
    )

    failed = len(results) - successful

    return respond(200, {
        "total_complaints": len(complaints),
        "successful": successful,
        "failed": failed,
        "results": results
    })
