import json
import boto3
import math
import re
from collections import Counter


TABLE_NAME = "campusresolve-tickets"

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)


# ---------------------------------------------------------
# TEXT TOKENIZATION
# ---------------------------------------------------------

def tokenize(text):

    if not text:
        return []

    text = str(text).lower()

    # Keep words only
    words = re.findall(r"[a-zA-Z0-9]+", text)

    # Remove very common words
    stop_words = {
        "the",
        "is",
        "in",
        "on",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "with",
        "my",
        "not"
    }

    return [
        word
        for word in words
        if word not in stop_words
    ]


# ---------------------------------------------------------
# TF
# ---------------------------------------------------------

def term_frequency(tokens):

    if not tokens:
        return {}

    counts = Counter(tokens)

    total = len(tokens)

    return {
        word: count / total
        for word, count in counts.items()
    }


# ---------------------------------------------------------
# IDF
# ---------------------------------------------------------

def inverse_document_frequency(documents):

    total_documents = len(documents)

    document_frequency = Counter()

    for document in documents:

        unique_words = set(document)

        for word in unique_words:

            document_frequency[word] += 1


    idf = {}

    for word, frequency in document_frequency.items():

        idf[word] = math.log(
            (total_documents + 1) /
            (frequency + 1)
        ) + 1

    return idf


# ---------------------------------------------------------
# TF-IDF VECTOR
# ---------------------------------------------------------

def tfidf_vector(tokens, idf):

    tf = term_frequency(tokens)

    return {
        word: value * idf.get(word, 0)
        for word, value in tf.items()
    }


# ---------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    if not vector_a or not vector_b:
        return 0.0

    common_words = (
        set(vector_a.keys())
        &
        set(vector_b.keys())
    )

    numerator = sum(
        vector_a[word] * vector_b[word]
        for word in common_words
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a.values()
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b.values()
        )
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return numerator / (
        magnitude_a * magnitude_b
    )


# ---------------------------------------------------------
# SCAN ALL TICKETS
# ---------------------------------------------------------

def get_all_tickets():

    items = []

    response = table.scan()

    items.extend(
        response.get("Items", [])
    )

    while "LastEvaluatedKey" in response:

        response = table.scan(
            ExclusiveStartKey=
            response["LastEvaluatedKey"]
        )

        items.extend(
            response.get("Items", [])
        )

    return items


# ---------------------------------------------------------
# IMPACT SCORE
# ---------------------------------------------------------

def calculate_impact_score(
    related_count,
    priority,
    needs_human_review,
    department_count
):

    # Related complaint contribution
    related_score = min(
        related_count * 5,
        40
    )

    # Priority contribution
    priority_scores = {
        "LOW": 5,
        "MEDIUM": 15,
        "HIGH": 25,
        "URGENT": 35
    }

    priority_score = priority_scores.get(
        str(priority).upper(),
        10
    )

    # Human review
    review_score = (
        15
        if needs_human_review
        else 0
    )

    # Department-wide impact
    department_score = min(
        department_count * 2,
        10
    )

    score = (
        related_score
        +
        priority_score
        +
        review_score
        +
        department_score
    )

    return min(
        round(score),
        100
    )


# ---------------------------------------------------------
# IMPACT LEVEL
# ---------------------------------------------------------

def impact_level(score):

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------
# MAIN HANDLER
# ---------------------------------------------------------

def lambda_handler(event, context):

    try:

        tickets = get_all_tickets()

        if not tickets:

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type":
                        "application/json",
                    "Access-Control-Allow-Origin":
                        "*"
                },
                "body": json.dumps({
                    "success": True,
                    "tickets": []
                })
            }


        # -------------------------------------------------
        # Prepare complaint documents
        # -------------------------------------------------

        documents = []

        for ticket in tickets:

            complaint = ticket.get(
                "complaint",
                ""
            )

            documents.append(
                tokenize(complaint)
            )


        # -------------------------------------------------
        # IDF
        # -------------------------------------------------

        idf = inverse_document_frequency(
            documents
        )


        # -------------------------------------------------
        # TF-IDF vectors
        # -------------------------------------------------

        vectors = [
            tfidf_vector(
                document,
                idf
            )
            for document in documents
        ]


        # -------------------------------------------------
        # Department counts
        # -------------------------------------------------

        department_counts = Counter(
            ticket.get(
                "department",
                "Unassigned"
            )
            for ticket in tickets
        )


        results = []


        # -------------------------------------------------
        # Compare every complaint
        # -------------------------------------------------

        for index, ticket in enumerate(tickets):

            current_vector = vectors[index]

            related = []


            for other_index, other_ticket in enumerate(tickets):

                if index == other_index:
                    continue


                similarity = cosine_similarity(
                    current_vector,
                    vectors[other_index]
                )


                # Similarity threshold
                if similarity >= 0.35:

                    related.append({
                        "ticket_id":
                            other_ticket.get(
                                "ticket_id"
                            ),

                        "similarity":
                            round(
                                similarity,
                                3
                            ),

                        "complaint":
                            other_ticket.get(
                                "complaint",
                                ""
                            )
                    })


            # Highest similarity first

            related.sort(
                key=lambda x:
                    x["similarity"],
                reverse=True
            )


            # Limit displayed related complaints

            related = related[:10]


            related_count = len(related)


            priority = ticket.get(
                "priority",
                ticket.get(
                    "priority_ml",
                    "MEDIUM"
                )
            )


            needs_review = ticket.get(
                "needs_human_review",
                False
            )


            department = ticket.get(
                "department",
                "Unassigned"
            )


            department_count = (
                department_counts.get(
                    department,
                    0
                )
            )


            score = calculate_impact_score(
                related_count,
                priority,
                needs_review,
                department_count
            )


            results.append({

                "ticket_id":
                    ticket.get(
                        "ticket_id"
                    ),

                "complaint":
                    ticket.get(
                        "complaint",
                        ""
                    ),

                "category":
                    ticket.get(
                        "category",
                        ""
                    ),

                "department":
                    department,

                "priority":
                    priority,

                "status":
                    ticket.get(
                        "status",
                        "OPEN"
                    ),

                "related_count":
                    related_count,

                "related_complaints":
                    related,

                "similarity_threshold":
                    0.35,

                "impact_score":
                    score,

                "impact_level":
                    impact_level(
                        score
                    ),

                "needs_human_review":
                    needs_review
            })


        return {

            "statusCode": 200,

            "headers": {

                "Content-Type":
                    "application/json",

                "Access-Control-Allow-Origin":
                    "*"
            },

            "body": json.dumps({

                "success": True,

                "total_tickets":
                    len(results),

                "results":
                    results

            })

        }


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return {

            "statusCode": 500,

            "headers": {

                "Content-Type":
                    "application/json",

                "Access-Control-Allow-Origin":
                    "*"
            },

            "body": json.dumps({

                "success": False,

                "error":
                    str(e)

            })

        }
