
import requests
import json

from config import (
    OLLAMA_URL,
    MODEL_NAME,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT
)


def analyze_review(
    review,
    temperature=0.2,
    max_tokens=500
):

    # System Prompt: Role / Personality
    system_prompt = """
    You are a professional university quality
    assurance analyst.

    Your job is to analyze student feedback objectively.

    Classify sentiment into exactly one of:
    Positive, Negative, Neutral, Mixed.

    Identify the main topics, issues, positive aspects,
    and practical recommendations.

    Do not invent information that is not present
    in the review.

    Sentiment classification rules:

- Positive: The review contains only positive
  feedback without meaningful criticism.

- Negative: The review contains only negative
  feedback without meaningful praise.

- Neutral: The review is factual or contains
  no clear positive or negative opinion.

- Mixed: The review contains both meaningful
  positive and negative feedback.

Important:
Evaluate the entire review, not only the final
sentence or the complaint after "but".

If both praise and criticism are present,
classify the sentiment as Mixed even if one
aspect seems more important.

    Return valid JSON only.
    """

    # Few-Shot Prompting
    user_prompt = f"""
    Follow these examples:

    Example 1:
    Review: "The professor explains very clearly."

    Output:
    {{
        "sentiment": "Positive",
        "topics": ["Teaching Quality"],
        "main_issue": "None",
        "positive_aspect": "Clear explanations",
        "recommendation": "Maintain the current teaching approach"
    }}

    Example 2:
    Review: "The exams are too difficult."

    Output:
    {{
        "sentiment": "Negative",
        "topics": ["Exam Difficulty"],
        "main_issue": "Exams are too difficult",
        "positive_aspect": "None",
        "recommendation": "Review exam difficulty"
    }}

    Now analyze this review:

    Review: {json.dumps(review)}

    Return a JSON object with exactly these fields:
    sentiment, topics, main_issue,
    positive_aspect, recommendation.
    """

    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        "format": "json",

        "stream": False,

        "options": {
    "temperature": float(temperature),
    "num_predict": int(max_tokens)
}
    }

    try:

        print(
    f"[OLLAMA REQUEST] "
    f"Temperature={payload['options']['temperature']} | "
    f"Max Tokens={payload['options']['num_predict']}",
    flush=True
)

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        content = result["message"]["content"]

        analysis = json.loads(content)

        # Validate AI output
        allowed_sentiments = [
            "Positive",
            "Negative",
            "Neutral",
            "Mixed"
        ]

        if analysis.get("sentiment") not in allowed_sentiments:
            raise ValueError("Invalid sentiment category")

        if not isinstance(analysis.get("topics"), list):
            raise ValueError("Topics must be a list")

        required_fields = [
            "main_issue",
            "positive_aspect",
            "recommendation"
        ]

        for field in required_fields:
            if not isinstance(analysis.get(field), str):
                raise ValueError(f"Invalid field: {field}")

        return analysis

    except requests.exceptions.Timeout:
        raise Exception("AI request timed out.")

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
        "Cannot connect to Ollama. "
        "Please start Ollama and try again."
    )

    except requests.exceptions.HTTPError:
        raise Exception(f"Ollama API error: {response.status_code}")

    except (KeyError, json.JSONDecodeError, ValueError) as e:
        raise Exception(f"Invalid AI response: {e}")


def analyze_reviews(
    reviews,
    progress_callback=None,
    temperature=0.2,
    max_tokens=500
):

    results = []

    total = len(reviews)

    for index, review in enumerate(reviews):

        try:

            analysis = analyze_review(
    review,
    temperature=temperature,
    max_tokens=max_tokens
)

            result = {
                "review": review,
                **analysis,
                "error": ""
            }

        except Exception as e:

            result = {
                "review": review,
                "sentiment": "Error",
                "topics": [],
                "main_issue": "Analysis failed",
                "positive_aspect": "Not available",
                "recommendation": "Please retry",
                "error": str(e)
            }

        results.append(result)

        if progress_callback:
            progress_callback((index + 1) / total)

    return results
