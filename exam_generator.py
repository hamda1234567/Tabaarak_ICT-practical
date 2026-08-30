import json

from rag.retreiver import retrieve_context
from llm.ollam_client import generate_response


def generate_exam(
    certification: str,
    difficulty: str,
    number_of_questions: int
):
    """
    Generate AI-900 practice exam questions
    using RAG + Ollama.
    """

    # ==========================================
    # 1. Validate certification
    # ==========================================

    certification = certification.strip().upper()

    if certification != "AI-900":
        raise ValueError(
            "Only AI-900 certification is supported."
        )

    # ==========================================
    # 2. Validate difficulty
    # ==========================================

    difficulty = difficulty.strip().lower()

    allowed_difficulties = [
        "easy",
        "medium",
        "hard"
    ]

    if difficulty not in allowed_difficulties:
        raise ValueError(
            "Difficulty must be Easy, Medium, or Hard."
        )

    # ==========================================
    # 3. Validate number of questions
    # ==========================================

    allowed_numbers = [5, 10, 20]

    if number_of_questions not in allowed_numbers:
        raise ValueError(
            "Number of questions must be 5, 10, or 20."
        )

    # ==========================================
    # 4. Retrieve context from RAG
    # ==========================================

    query = f"""
    Microsoft AI-900 certification.

    Difficulty: {difficulty}

    Important topics:
    - Machine Learning
    - Computer Vision
    - Natural Language Processing
    - Generative AI
    - Responsible AI
    - Azure AI Services

    Find relevant knowledge for creating
    certification practice questions.
    """

    context = retrieve_context(
        query=query,
        top_k=5
    )

    if not context:
        raise ValueError(
            "No knowledge found in the RAG database."
        )

    # ==========================================
    # 5. Create prompt for Qwen
    # ==========================================

    prompt = f"""
You are an expert Microsoft AI-900
certification practice exam generator.

Create exactly {number_of_questions}
multiple-choice questions.

Certification:
AI-900

Difficulty:
{difficulty}

Use ONLY the knowledge provided below.

RULES:

1. Generate exactly {number_of_questions} questions.
2. Every question must have exactly 4 options.
3. Each question must have ONE correct answer.
4. Questions must be original.
5. Do not copy the source text as questions.
6. Include a topic for every question.
7. Return ONLY valid JSON.
8. Do not use Markdown.
9. Do not use ```json.
10. Do not add explanations outside the JSON.

Use this exact JSON structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "correct_answer": "Option A",
            "topic": "Computer Vision"
        }}
    ]
}}

KNOWLEDGE CONTEXT:
==================

{context}

==================
"""

    # ==========================================
    # 6. Send prompt to Ollama
    # ==========================================

    response = generate_response(prompt)

    if response.startswith("ERROR:"):
        raise RuntimeError(response)

    # ==========================================
    # 7. Clean Qwen response
    # ==========================================

    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    # ==========================================
    # 8. Convert response to JSON
    # ==========================================

    try:

        data = json.loads(response)

    except json.JSONDecodeError as error:

        raise ValueError(
            "Qwen returned invalid JSON.\n\n"
            f"JSON Error: {error}\n\n"
            f"Qwen Response:\n{response}"
        )

    # ==========================================
    # 9. Check questions field
    # ==========================================

    if "questions" not in data:

        raise ValueError(
            "Response does not contain 'questions'."
        )

    questions = data["questions"]

    if not isinstance(questions, list):

        raise ValueError(
            "'questions' must be a list."
        )

    # ==========================================
    # 10. Check question count
    # ==========================================

    if len(questions) != number_of_questions:

        raise ValueError(
            f"Expected {number_of_questions} questions, "
            f"but received {len(questions)}."
        )

    # ==========================================
    # 11. Validate every question
    # ==========================================

    for index, question in enumerate(questions):

        required_fields = [
            "question",
            "options",
            "correct_answer",
            "topic"
        ]

        for field in required_fields:

            if field not in question:

                raise ValueError(
                    f"Question {index + 1} "
                    f"is missing '{field}'."
                )

        # Check options

        options = question["options"]

        if not isinstance(options, list):

            raise ValueError(
                f"Question {index + 1} "
                f"options must be a list."
            )

        if len(options) != 4:

            raise ValueError(
                f"Question {index + 1} "
                f"must have exactly 4 options."
            )

        # Check correct answer

        correct_answer = question["correct_answer"]

        if correct_answer not in options:

            raise ValueError(
                f"Question {index + 1} "
                f"has an invalid correct answer."
            )

    # ==========================================
    # 12. Return final exam
    # ==========================================

    return {
        "certification": certification,
        "difficulty": difficulty,
        "number_of_questions": number_of_questions,
        "questions": questions
    }