
import json
import random
import re
from pathlib import Path
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Certification Assessment Assistant",
    description="AI-900 Practice Exam Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_FILE = (
    BASE_DIR
    / "data"
    / "certification_knowledge"
    / "a1_900.txt"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_knowledge():
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    return KNOWLEDGE_FILE.read_text(
        encoding="utf-8"
    )


knowledge = load_knowledge()


# ============================================================
# TOPICS
# ============================================================

TOPICS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Speech",
    "Generative AI",
    "Responsible AI",
    "Azure AI Services"
]


# ============================================================
# DATA MODELS
# ============================================================

class ExamRequest(BaseModel):
    difficulty: str = "medium"
    question_count: int = 10


class Answer(BaseModel):
    question_id: int
    answer: str


class ExamSubmission(BaseModel):
    answers: List[Answer]


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt: str):

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "")

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. "
                "Please start Ollama first."
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error: {str(e)}"
        )


# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def extract_json(text: str):

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json|```",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    # Find JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_questions(
    difficulty: str,
    question_count: int
):

    # Limit knowledge sent to model
    context = knowledge[:18000]

    prompt = f"""
You are an AI-900 Microsoft Azure AI Fundamentals exam question generator.

Use ONLY the certification knowledge provided below.

CERTIFICATION:
Microsoft Azure AI Fundamentals (AI-900)

DIFFICULTY:
{difficulty}

NUMBER OF QUESTIONS:
{question_count}

KNOWLEDGE:
{context}

Generate exactly {question_count} multiple-choice questions.

Each question must have:
- id
- topic
- question
- 4 options
- correct_answer

Use this exact JSON format:

{{
  "questions": [
    {{
      "id": 1,
      "topic": "Machine Learning",
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct_answer": "Option B"
    }}
  ]
}}

IMPORTANT:
- Exactly 4 options per question.
- Only one correct answer.
- Do not explain the answer.
- Do not add text outside JSON.
- Questions must be based on the supplied knowledge.
"""

    result = ask_ollama(prompt)

    data = extract_json(result)

    if not data or "questions" not in data:
        raise HTTPException(
            status_code=500,
            detail="Could not generate valid questions from Ollama."
        )

    questions = data["questions"]

    # Make IDs consistent
    for index, question in enumerate(questions, start=1):
        question["id"] = index

    return questions


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Certification Assessment Assistant API is running!",
        "status": "success",
        "certification": "Microsoft Azure AI Fundamentals (AI-900)",
        "ollama_model": OLLAMA_MODEL
    }


# ============================================================
# KNOWLEDGE
# ============================================================

@app.get("/knowledge")
def get_knowledge():

    return {
        "file": str(KNOWLEDGE_FILE),
        "characters": len(knowledge),
        "topics": TOPICS
    }


# ============================================================
# GENERATE EXAM
# ============================================================

@app.post("/generate-exam")
def generate_exam(request: ExamRequest):

    if request.question_count not in [5, 10, 20]:
        raise HTTPException(
            status_code=400,
            detail="Question count must be 5, 10, or 20."
        )

    difficulty = request.difficulty.lower()

    if difficulty not in [
        "easy",
        "medium",
        "hard"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Difficulty must be easy, medium, or hard."
        )

    questions = generate_questions(
        difficulty,
        request.question_count
    )

    # Store answers temporarily in memory
    global current_exam
    current_exam = questions

    # Don't send correct answers to frontend
    safe_questions = []

    for question in questions:

        safe_questions.append({
            "id": question["id"],
            "topic": question.get(
                "topic",
                "AI Fundamentals"
            ),
            "question": question["question"],
            "options": question["options"]
        })

    return {
        "certification": "Microsoft Azure AI Fundamentals (AI-900)",
        "difficulty": difficulty,
        "question_count": len(safe_questions),
        "questions": safe_questions
    }


# ============================================================
# CURRENT QUESTIONS
# ============================================================

current_exam = []


@app.get("/questions")
def get_questions():

    if not current_exam:

        return {
            "message": "No exam generated yet.",
            "questions": []
        }

    safe_questions = []

    for question in current_exam:

        safe_questions.append({
            "id": question["id"],
            "topic": question.get(
                "topic",
                "AI Fundamentals"
            ),
            "question": question["question"],
            "options": question["options"]
        })

    return {
        "total_questions": len(safe_questions),
        "questions": safe_questions
    }


# ============================================================
# SUBMIT EXAM
# ============================================================

@app.post("/submit-exam")
def submit_exam(submission: ExamSubmission):

    if not current_exam:
        raise HTTPException(
            status_code=400,
            detail="No exam is currently available."
        )

    score = 0
    results = []
    topic_scores = {}

    for submitted in submission.answers:

        question = next(
            (
                q for q in current_exam
                if q["id"] == submitted.question_id
            ),
            None
        )

        if question is None:
            continue

        correct_answer = question["correct_answer"]

        is_correct = (
            submitted.answer.strip().lower()
            == correct_answer.strip().lower()
        )

        if is_correct:
            score += 1

        topic = question.get(
            "topic",
            "AI Fundamentals"
        )

        if topic not in topic_scores:
            topic_scores[topic] = {
                "correct": 0,
                "total": 0
            }

        topic_scores[topic]["total"] += 1

        if is_correct:
            topic_scores[topic]["correct"] += 1

        results.append({
            "question_id": submitted.question_id,
            "topic": topic,
            "your_answer": submitted.answer,
            "correct_answer": correct_answer,
            "correct": is_correct
        })

    total = len(submission.answers)

    percentage = (
        round((score / total) * 100, 2)
        if total > 0
        else 0
    )

    if percentage >= 80:
        status = "Excellent"
    elif percentage >= 70:
        status = "Pass"
    elif percentage >= 50:
        status = "Needs Improvement"
    else:
        status = "Fail"

    # ========================================================
    # WEAK TOPICS
    # ========================================================

    weak_topics = []

    for topic, data in topic_scores.items():

        topic_percentage = (
            data["correct"] / data["total"]
        ) * 100

        if topic_percentage < 70:

            weak_topics.append({
                "topic": topic,
                "percentage": round(
                    topic_percentage,
                    2
                )
            })

    return {
        "score": score,
        "total_questions": total,
        "percentage": percentage,
        "status": status,
        "weak_topics": weak_topics,
        "topic_scores": topic_scores,
        "results": results
    }


# ============================================================
# STUDY PLAN
# ============================================================

@app.get("/study-plan")
def get_study_plan(
    score: int = 0
):

    if score >= 80:

        level = "Advanced"

        plan = [
            {
                "day": 1,
                "topic": "Machine Learning",
                "task": "Review classification, regression and clustering."
            },
            {
                "day": 2,
                "topic": "Computer Vision",
                "task": "Review image classification, object detection and OCR."
            },
            {
                "day": 3,
                "topic": "NLP",
                "task": "Review sentiment analysis, translation and language detection."
            },
            {
                "day": 4,
                "topic": "Generative AI",
                "task": "Review LLMs, prompts and generated content."
            },
            {
                "day": 5,
                "topic": "Responsible AI",
                "task": "Review fairness, privacy, transparency and accountability."
            }
        ]

    elif score >= 50:

        level = "Intermediate"

        plan = [
            {
                "day": 1,
                "topic": "AI Fundamentals",
                "task": "Review common AI workloads."
            },
            {
                "day": 2,
                "topic": "Machine Learning",
                "task": "Study supervised and unsupervised learning."
            },
            {
                "day": 3,
                "topic": "Computer Vision",
                "task": "Study image classification, object detection and OCR."
            },
            {
                "day": 4,
                "topic": "NLP and Speech",
                "task": "Review NLP and speech capabilities."
            },
            {
                "day": 5,
                "topic": "Generative AI",
                "task": "Review LLMs and prompt engineering."
            }
        ]

    else:

        level = "Beginner"

        plan = [
            {
                "day": 1,
                "topic": "Artificial Intelligence",
                "task": "Learn AI fundamentals and common AI workloads."
            },
            {
                "day": 2,
                "topic": "Machine Learning",
                "task": "Learn classification, regression and clustering."
            },
            {
                "day": 3,
                "topic": "Computer Vision",
                "task": "Learn image classification, object detection and OCR."
            },
            {
                "day": 4,
                "topic": "NLP and Speech",
                "task": "Learn NLP, speech-to-text and text-to-speech."
            },
            {
                "day": 5,
                "topic": "Responsible AI",
                "task": "Learn fairness, safety, privacy, transparency and accountability."
            }
        ]

    return {
        "score": score,
        "level": level,
        "study_plan": plan
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "backend": "running",
        "ollama": OLLAMA_MODEL,
        "knowledge_file": KNOWLEDGE_FILE.exists(),
        "knowledge_characters": len(knowledge)
    }
