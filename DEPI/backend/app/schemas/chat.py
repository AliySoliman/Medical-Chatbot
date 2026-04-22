# backend/app/schemas/chat.py
# ─────────────────────────────────────────────────────────────────────────────
# MedCortex Chat Schemas
# Pydantic models for the /chat endpoint request and response
# ─────────────────────────────────────────────────────────────────────────────

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's health-related message or symptom description",
        example="I have a fever, sore throat, and body aches since yesterday",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID for personalisation (future use)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUB-MODELS
# ─────────────────────────────────────────────────────────────────────────────
class Source(BaseModel):
    book:    str
    section: str


class LifestyleRecommendations(BaseModel):
    foods_to_eat:           List[str] = []
    foods_to_avoid:         List[str] = []
    drinks_to_have:         List[str] = []
    drinks_to_avoid:        List[str] = []
    exercises_recommended:  List[str] = []
    exercises_to_avoid:     List[str] = []
    rest_recommendation:    str       = ""


class Doctor(BaseModel):
    name:      str
    specialty: str
    address:   str
    phone:     str
    npi:       str
    source:    str = "Doctor lookup source"


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────────────────────────────────────────
class ChatResponse(BaseModel):
    # Core diagnosis
    answer:               str         = Field(..., description="Full clinical response from the RAG")
    suspected_conditions: List[str]   = Field(default=[], description="Extracted suspected diagnoses")
    symptoms:             List[str]   = Field(default=[], description="Symptoms extracted from user message")
    sources:              List[Source] = Field(default=[], description="Medical textbook sources used")

    # Recommendations
    recommendations: LifestyleRecommendations = Field(
        default_factory=LifestyleRecommendations,
        description="Food, drink, exercise, and rest recommendations",
    )

    # Doctors
    doctors: List[Doctor] = Field(
        default=[],
        description="Relevant doctors and clinics from all configured lookup sources",
    )
