# backend/app/rag/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# MedCortex RAG Pipeline
# Connects: Pinecone (medical-assistant) → BGE-Large → Groq Llama 3.3 70B
# Namespace: medical_textbooks_base
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
from functools import lru_cache
from typing import List, Dict, Any

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — match exactly what the notebook used
# ─────────────────────────────────────────────────────────────────────────────
PINECONE_INDEX   = "medical-assistant"
PINECONE_NS      = "medical_textbooks_base"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
LLM_MODEL        = "llama-3.3-70b-versatile"
RETRIEVER_K      = 5
RETRIEVER_FETCH  = 12

SYMPTOM_TRIGGER_TERMS = (
    "pain", "ache", "aching", "hurt", "hurts", "burning", "discomfort",
    "swelling", "fever", "bleeding", "vomiting", "nausea", "dizziness",
    "fatigue", "weakness", "shortness of breath", "cough", "rash",
    "headache", "migraine", "back pain", "chest pain", "stomach pain",
    "cramp", "cramps", "tingling", "numbness", "infection", "injury",
    "sore", "soreness", "pressure", "palpitations",
)

DISTRESS_TRIGGER_TERMS = (
    "disturbing", "terrible", "unbearable", "awful", "severe", "intense",
    "extreme", "scary", "frightening", "worrying", "worried", "concerned",
)

REFERRAL_BLOCK_REGEX = re.compile(r"\[DOCTOR_REFERRAL\](.*?)\[/DOCTOR_REFERRAL\]", re.DOTALL)

SPECIALIST_RULES = [
    (("back pain", "lower back", "upper back", "neck pain", "spine", "joint pain", "knee", "shoulder"), "orthopedic surgeon"),
    (("chest pain", "palpitations", "heart", "blood pressure"), "cardiologist"),
    (("rash", "itching", "skin", "mole", "acne"), "dermatologist"),
    (("headache", "migraine", "numbness", "tingling", "seizure", "dizziness"), "neurologist"),
    (("ear", "nose", "throat", "sinus"), "ENT specialist"),
    (("stomach", "abdomen", "abdominal", "nausea", "vomiting", "diarrhea", "constipation", "reflux"), "gastroenterologist"),
    (("cough", "shortness of breath", "wheezing", "breathing"), "pulmonologist"),
    (("burning urination", "urine", "kidney"), "urologist"),
    (("anxiety", "depression", "panic", "mental"), "psychiatrist"),
]

IMPORTANT_MEDICAL_PHRASES = (
    "lower back", "upper back", "back pain", "chest pain", "shortness of breath",
    "stomach pain", "abdominal pain", "burning urination", "joint pain",
    "severe pain", "disturbing pain", "neck pain", "headache", "skin rash",
)

IMPORTANT_MEDICAL_WORDS = {
    "pain", "ache", "aching", "hurt", "hurts", "burning", "discomfort", "swelling",
    "fever", "bleeding", "vomiting", "nausea", "dizziness", "fatigue", "weakness",
    "cough", "rash", "headache", "migraine", "cramp", "cramps", "tingling",
    "numbness", "infection", "injury", "sore", "soreness", "pressure", "palpitations",
    "back", "lower", "upper", "neck", "spine", "joint", "knee", "shoulder", "hip",
    "chest", "heart", "stomach", "abdomen", "abdominal", "ear", "nose", "throat",
    "sinus", "breathing", "breath", "urination", "urine", "kidney", "skin",
    "itching", "mole", "acne", "disturbing", "terrible", "unbearable", "awful",
    "severe", "intense", "extreme",
}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON LOADERS  (loaded once, reused across requests)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load BGE-Large once and cache it. Uses GPU if available."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MedCortex] Loading embeddings on {device}...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> PineconeVectorStore:
    """Connect to existing Pinecone index — does NOT re-upload anything."""
    return PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=_get_embeddings(),
        namespace=PINECONE_NS,
    )


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    """Instantiate Groq Llama 3.3 70B once."""
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=1024,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def extract_symptoms(user_message: str) -> List[str]:
    """
    Uses Llama to extract a clean list of medical symptoms from free-form text.
    Returns: ["fever", "sore throat", "body aches"]
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a medical symptom extractor. "
         "Extract ONLY the medical symptoms from the user message. "
         "Return a JSON array of symptom strings. "
         "Example output: [\"fever\", \"headache\", \"fatigue\"] "
         "Return ONLY the JSON array — no explanation, no markdown."),
        ("human", "{message}"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"message": user_message})

    # Safely parse the JSON array
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        symptoms = json.loads(cleaned)
        return [s.lower() for s in symptoms if isinstance(s, str)]
    except Exception:
        # Fallback: split on commas if JSON parse fails
        return [s.strip().strip('"').lower() for s in raw.split(",") if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE FORMATTER
# ─────────────────────────────────────────────────────────────────────────────
def _format_docs(docs) -> str:
    """Format retrieved chunks with their source metadata for the LLM context."""
    chunks = []
    for doc in docs:
        book    = doc.metadata.get("book_title", "Unknown Book")
        heading = doc.metadata.get("docling_headings", "Unknown Section")
        text    = doc.page_content
        chunks.append(f"Source: {book} | Section: {heading}\nText: {text}")
    return "\n\n---\n\n".join(chunks)


def should_force_doctor_referral(user_message: str) -> bool:
    """Deterministically flag messages that should reinforce doctor-referral output."""
    normalized = user_message.lower()
    return any(term in normalized for term in (*SYMPTOM_TRIGGER_TERMS, *DISTRESS_TRIGGER_TERMS))


def normalize_medical_message(user_message: str) -> str:
    """
    Build a more clinical representation of the user's message by preserving the
    medically meaningful phrases and words while stripping conversational filler.
    This is intentionally deterministic rather than model-based.
    """
    normalized = user_message.lower()
    captured_phrases = [phrase for phrase in IMPORTANT_MEDICAL_PHRASES if phrase in normalized]
    word_matches = re.findall(r"[a-z']+", normalized)
    captured_words = [word for word in word_matches if word in IMPORTANT_MEDICAL_WORDS]

    deduped_terms: List[str] = []
    for term in [*captured_phrases, *captured_words]:
        if term not in deduped_terms:
            deduped_terms.append(term)

    if not deduped_terms:
        return user_message

    return f"{user_message}\n\nClinical focus terms: {', '.join(deduped_terms[:18])}"


def build_rag_input(user_message: str) -> str:
    """
    Append an internal instruction for symptom-heavy messages so the LLM sees the
    referral requirement right next to the user message.
    """
    if not should_force_doctor_referral(user_message):
        return user_message

    return (
        f"{user_message}\n\n"
        "[System note: This message contains symptom descriptions or distress language. "
        "CRITICAL RULE — NO EXCEPTIONS: any message containing pain, ache, hurt, burning, "
        "discomfort, swelling, or another physical symptom description MUST include exactly one "
        "DOCTOR_REFERRAL block if a medical professional visit may be needed. Emotional language "
        "does not override this rule.]"
    )


def extract_doctor_referral_block(answer_text: str) -> str | None:
    match = REFERRAL_BLOCK_REGEX.search(answer_text)
    return match.group(0) if match else None


def infer_referral_specialist(symptoms: List[str], suspected_conditions: List[str], user_message: str) -> str:
    haystack = " ".join([*symptoms, *suspected_conditions, user_message.lower()])
    for keywords, specialist in SPECIALIST_RULES:
        if any(keyword in haystack for keyword in keywords):
            return specialist
    return "general practitioner"


def infer_referral_urgency(symptoms: List[str], user_message: str) -> str:
    haystack = " ".join([*symptoms, user_message.lower()])

    urgent_terms = (
        "chest pain", "shortness of breath", "trouble breathing", "severe bleeding",
        "loss of consciousness", "fainting", "stroke", "seizure",
    )
    soon_terms = (
        "severe", "unbearable", "disturbing", "extreme", "back pain", "lower back",
        "numbness", "tingling", "vomiting", "fever", "injury",
    )

    if any(term in haystack for term in urgent_terms):
        return "urgent"
    if any(term in haystack for term in soon_terms):
        return "soon"
    return "routine"


def infer_referral_reason(symptoms: List[str], user_message: str) -> str:
    if symptoms:
        primary = symptoms[0].strip().lower()
        return " ".join(primary.split()[:5])

    words = re.findall(r"[a-zA-Z]+", user_message.lower())
    return " ".join(words[:5]) or "physical symptoms"


def build_fallback_referral_block(
    symptoms: List[str],
    suspected_conditions: List[str],
    user_message: str,
) -> str | None:
    if not (symptoms or should_force_doctor_referral(user_message)):
        return None

    payload = {
        "specialist": infer_referral_specialist(symptoms, suspected_conditions, user_message),
        "urgency": infer_referral_urgency(symptoms, user_message),
        "reason": infer_referral_reason(symptoms, user_message),
    }
    return f'[DOCTOR_REFERRAL]{json.dumps(payload)}[/DOCTOR_REFERRAL]'


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RAG CHAIN
# ─────────────────────────────────────────────────────────────────────────────
def _build_rag_chain():
    """Build the LangChain RAG chain (MMR retriever → prompt → Llama → string)."""
    retriever = _get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH},
    )

    system_prompt = (
        "You are MedCortex, an elite Clinical AI Assistant. "
        "Your knowledge is strictly limited to the provided medical textbook excerpts.\n\n"
        "RULES:\n"
        "1. Answer ONLY from the context below. Do NOT hallucinate.\n"
        "2. If the answer is not in the context, say: "
        "'I cannot find the answer to this in my medical library.'\n"
        "3. Structure your answer clearly using bullet points where appropriate.\n"
        "4. End your response with a JSON block on its own line in this exact format:\n"
        "   SUSPECTED_CONDITIONS: [\"Condition1\", \"Condition2\"]\n"
        "5. CRITICAL RULE — NO EXCEPTIONS: any message containing pain, ache, hurt, burning, "
        "discomfort, swelling, or any physical symptom description MUST trigger the doctor "
        "referral JSON block when a doctor visit may be needed. Emotional language does not "
        "override this rule.\n"
        "6. This includes messages where the user expresses distress, uses words like "
        "\"disturbing\", \"terrible\", \"unbearable\", \"awful\", or \"severe\", or describes "
        "any pain regardless of how it is phrased.\n"
        "7. When the rule above applies, you MUST include the following JSON block somewhere in "
        "your response, on its own line, with no surrounding text on that line:\n\n"
        "[DOCTOR_REFERRAL]{{\"specialist\":\"<specialist type>\",\"urgency\":\"<routine|soon|urgent>\","
        "\"reason\":\"<one short phrase>\"}}[/DOCTOR_REFERRAL]\n\n"
        "Rules for the JSON block:\n"
        "- \"specialist\" must be a plain English doctor type suitable for a Google Places search, "
        "for example: \"orthopedic surgeon\", \"cardiologist\", \"dermatologist\", "
        "\"physical therapist\", \"neurologist\", \"general practitioner\", "
        "\"ENT specialist\", \"gastroenterologist\". Use \"general practitioner\" when unsure.\n"
        "- \"urgency\" must be exactly one of: \"routine\", \"soon\", \"urgent\".\n"
        "- \"reason\" must be 5 words or fewer describing why, for example: "
        "\"neck and back pain\".\n"
        "- Do NOT include this block for general health questions, medication questions, "
        "or when no doctor visit is needed.\n"
        "- Do NOT include this block more than once per response.\n\n"
        "MEDICAL CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  "{input}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "input": RunnablePassthrough()}
        | prompt
        | _get_llm()
        | StrOutputParser()
    )
    return chain, retriever


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def run_rag(user_message: str) -> Dict[str, Any]:
    """
    Main entry point called by routes/chat.py.

    Returns:
    {
        "answer":               str,        # Full clinical response text
        "suspected_conditions": List[str],  # e.g. ["Influenza", "Common Cold"]
        "symptoms":             List[str],  # extracted symptom list
        "sources":              List[dict], # book title + section for each chunk
    }
    """
    # 1. Extract symptoms
    normalized_message = normalize_medical_message(user_message)
    symptoms = extract_symptoms(normalized_message)

    # 2. Run RAG
    chain, retriever = _build_rag_chain()
    rag_input = build_rag_input(normalized_message)

    # Retrieve source docs separately so we can return them
    source_docs = retriever.invoke(normalized_message)
    answer_raw  = chain.invoke(rag_input)

    # 3. Parse suspected conditions from the answer
    suspected = []
    match = re.search(r'SUSPECTED_CONDITIONS:\s*(\[.*?\])', answer_raw)
    if match:
        try:
            suspected = json.loads(match.group(1))
        except Exception:
            pass

    if not extract_doctor_referral_block(answer_raw):
        fallback_referral = build_fallback_referral_block(symptoms, suspected, normalized_message)
        if fallback_referral:
            answer_raw = f"{answer_raw.rstrip()}\n{fallback_referral}"

    # Clean the answer (remove the JSON line from display text)
    answer_clean = re.sub(r'SUSPECTED_CONDITIONS:.*', '', answer_raw).strip()

    # 4. Build sources list
    sources = []
    seen    = set()
    for doc in source_docs:
        book = doc.metadata.get("book_title", "Unknown")
        sec  = doc.metadata.get("docling_headings", "")
        key  = f"{book}|{sec}"
        if key not in seen:
            seen.add(key)
            sources.append({"book": book, "section": sec})

    return {
        "answer":               answer_clean,
        "suspected_conditions": suspected,
        "symptoms":             symptoms,
        "sources":              sources,
    }
