// frontend/services/chat.ts
// ─────────────────────────────────────────────────────────────────────────────
// MedCortex Chat Service
// Handles all API calls to the FastAPI /chat endpoint
// ─────────────────────────────────────────────────────────────────────────────

import axios from "axios";
import type { DoctorReferral } from "@/lib/extractDoctorReferral";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES — mirror the FastAPI ChatResponse schema exactly
// ─────────────────────────────────────────────────────────────────────────────
export interface Source {
  book: string;
  section: string;
}

export interface LifestyleRecommendations {
  foods_to_eat:           string[];
  foods_to_avoid:         string[];
  drinks_to_have:         string[];
  drinks_to_avoid:        string[];
  exercises_recommended:  string[];
  exercises_to_avoid:     string[];
  rest_recommendation:    string;
}

export interface Doctor {
  name:      string;
  specialty: string;
  address:   string;
  phone:     string;
  npi:       string;
  source:    string;
}

export interface ChatResponse {
  answer:               string;
  suspected_conditions: string[];
  symptoms:             string[];
  sources:              Source[];
  recommendations:      LifestyleRecommendations;
  doctors:              Doctor[];
}

export interface ChatMessage {
  id:        string;
  role:      "user" | "assistant";
  content:   string;
  data?:     ChatResponse;   // only on assistant messages
  doctorReferral?: DoctorReferral | null;
  timestamp: Date;
}

export interface ChatThread {
  id:        string;
  title:     string;
  messages:  ChatMessage[];
  updatedAt: number;
  pinned?:   boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// API CALL
// ─────────────────────────────────────────────────────────────────────────────
export async function sendMessage(message: string): Promise<ChatResponse> {
  const token = localStorage.getItem("token");

  const response = await axios.post<ChatResponse>(
    `${API_BASE}/chat`,
    { message },
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );

  return response.data;
}
