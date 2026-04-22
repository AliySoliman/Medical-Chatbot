# backend/app/routes/chat.py
# ─────────────────────────────────────────────────────────────────────────────
# MedCortex Chat Route
# POST /chat  →  RAG diagnosis + lifestyle recommendations + hybrid doctor finder
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
from fastapi import APIRouter, HTTPException, status
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    LifestyleRecommendations,
    Doctor,
    Source,
)
from app.rag.pipeline           import run_rag
from app.recommenders.lifestyle_model import get_lifestyle_recommendations
from app.recommenders.doctor_model    import find_doctors

router = APIRouter(prefix="/chat", tags=["Chat"])


SPECIALTY_FALLBACK_RULES = [
    (("back pain", "lower back", "upper back", "neck pain", "spine", "joint pain", "knee", "shoulder"), "Orthopedic Surgeon"),
    (("chest pain", "palpitations", "heart", "blood pressure"), "Cardiologist"),
    (("rash", "itching", "skin", "mole", "acne"), "Dermatologist"),
    (("headache", "migraine", "numbness", "tingling", "seizure", "dizziness"), "Neurologist"),
    (("ear", "nose", "throat", "sinus"), "ENT Specialist"),
    (("stomach", "abdomen", "abdominal", "nausea", "vomiting", "diarrhea", "constipation", "reflux"), "Gastroenterologist"),
    (("cough", "shortness of breath", "wheezing", "breathing"), "Pulmonologist"),
    (("burning urination", "urine", "kidney"), "Urologist"),
]


def infer_doctor_specialties(symptoms: list[str], suspected_conditions: list[str], message: str) -> list[str]:
    haystack = " ".join([*symptoms, *suspected_conditions, message.lower()])
    specialties: list[str] = []

    for keywords, specialty in SPECIALTY_FALLBACK_RULES:
        if any(keyword in haystack for keyword in keywords) and specialty not in specialties:
            specialties.append(specialty)

    if specialties:
        return specialties[:2]

    if symptoms:
        return ["General Practitioner"]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a health message and receive diagnosis + recommendations",
    status_code=status.HTTP_200_OK,
)
async def chat(request: ChatRequest):
    """
    Full MedCortex pipeline:

    1. RAG  →  clinical answer + suspected conditions + symptoms + sources
    2. Lifestyle LLM  →  food / drink / exercise JSON
    3. Hybrid lookup  →  Egypt clinics/doctors + existing foreign doctors by specialty

    Returns a unified ChatResponse.
    """
    try:
        # ── Step 1: RAG (runs in thread pool to avoid blocking the event loop) ──
        loop = asyncio.get_running_loop()
        rag_result = await loop.run_in_executor(None, run_rag, request.message)

        suspected_conditions = rag_result.get("suspected_conditions", [])
        symptoms             = rag_result.get("symptoms", [])
        answer               = rag_result.get("answer", "")
        raw_sources          = rag_result.get("sources", [])

        # ── Step 2 & 3: lifestyle recommendations then doctor lookup ─────────────
        lifestyle_future = loop.run_in_executor(
            None,
            get_lifestyle_recommendations,
            suspected_conditions,
            symptoms,
        )
        # We need specialties from lifestyle first, so run step 2 then step 3
        lifestyle_raw = await lifestyle_future

        doctor_specialties = lifestyle_raw.get("doctor_specialties", [])
        if not doctor_specialties:
            doctor_specialties = infer_doctor_specialties(
                symptoms,
                suspected_conditions,
                request.message,
            )
        doctors_raw = await loop.run_in_executor(
            None, find_doctors, doctor_specialties
        )

        # ── Assemble response ──────────────────────────────────────────────────
        sources = [Source(**s) for s in raw_sources]

        recommendations = LifestyleRecommendations(
            foods_to_eat=          lifestyle_raw.get("foods_to_eat",          []),
            foods_to_avoid=        lifestyle_raw.get("foods_to_avoid",        []),
            drinks_to_have=        lifestyle_raw.get("drinks_to_have",        []),
            drinks_to_avoid=       lifestyle_raw.get("drinks_to_avoid",       []),
            exercises_recommended= lifestyle_raw.get("exercises_recommended", []),
            exercises_to_avoid=    lifestyle_raw.get("exercises_to_avoid",    []),
            rest_recommendation=   lifestyle_raw.get("rest_recommendation",   ""),
        )

        doctors = [Doctor(**d) for d in doctors_raw]

        return ChatResponse(
            answer=               answer,
            suspected_conditions= suspected_conditions,
            symptoms=             symptoms,
            sources=              sources,
            recommendations=      recommendations,
            doctors=              doctors,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MedCortex pipeline error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# GET /chat/health  (simple liveness check)
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health", summary="Check chat service is alive")
async def health():
    return {"status": "ok", "service": "MedCortex Chat"}
