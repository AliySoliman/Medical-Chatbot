"use client";

import { useState } from "react";
import { useDoctorRecommendations, type Doctor as RecommendedDoctor } from "@/hooks/useDoctorRecommendations";
import type { DoctorReferral } from "@/lib/extractDoctorReferral";
import type { ChatResponse, Doctor as ApiDoctor, Source } from "@/services/chat";

interface ResponseCardProps {
  data: ChatResponse;
  referral?: DoctorReferral | null;
  coordinates?: { lat: number; lng: number } | null;
}

type Tab = "diagnosis" | "lifestyle" | "doctors" | "sources";

type DisplayDoctor = {
  name: string;
  specialty: string;
  address: string;
  phone: string | null;
  mapsUrl: string | null;
  rating: number | null;
  reviewCount: number;
  badgeLabel: string;
  badgeTone: string;
  sourceText: string;
  reference: string | null;
};

const EGYPT_LOCATION_REGEX =
  /\b(egypt|cairo|giza|alexandria|alexandrie|dakahlia|mansoura|tanta|zagazig|aswan|luxor|sohag|minya|assiut|faiyum|fayoum|ismailia|port said|suez|beni suef|qena|damietta|monufia|menoufia|beheira|kafr el-sheikh|kafr el sheikh|gharbia|sharqia)\b/i;

function buildMapsUrl(name: string, address: string) {
  const query = [name, address].filter(Boolean).join(", ");
  return query
    ? `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`
    : null;
}

function isEgyptDoctor(address: string, sourceText: string) {
  return /egypt/i.test(sourceText) || EGYPT_LOCATION_REGEX.test(address);
}

function mapApiDoctorToDisplay(doctor: ApiDoctor): DisplayDoctor {
  const egyptDoctor = isEgyptDoctor(doctor.address, doctor.source);

  return {
    name: doctor.name,
    specialty: doctor.specialty,
    address: doctor.address,
    phone: doctor.phone || null,
    mapsUrl: buildMapsUrl(doctor.name, doctor.address),
    rating: null,
    reviewCount: 0,
    badgeLabel: egyptDoctor ? "Egypt doctor/clinic" : "Foreign doctor",
    badgeTone: egyptDoctor
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : "border-sky-200 bg-sky-50 text-sky-700",
    sourceText: doctor.source,
    reference: doctor.npi ? `Ref: ${doctor.npi.slice(-6)}` : null,
  };
}

function mapRecommendedDoctorToDisplay(doctor: RecommendedDoctor): DisplayDoctor {
  const egyptDoctor = doctor.source === "places" || isEgyptDoctor(doctor.address, doctor.sourceLabel);

  return {
    name: doctor.name,
    specialty: doctor.specialty || "Doctor",
    address: doctor.address,
    phone: doctor.phone || null,
    mapsUrl: doctor.mapsUrl ?? buildMapsUrl(doctor.name, doctor.address),
    rating: doctor.rating,
    reviewCount: doctor.reviewCount,
    badgeLabel: egyptDoctor ? "Egypt doctor/clinic" : "Foreign doctor",
    badgeTone: egyptDoctor
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : "border-sky-200 bg-sky-50 text-sky-700",
    sourceText: doctor.source === "places" ? "Google Maps nearby search" : doctor.sourceLabel,
    reference: null,
  };
}

export default function ResponseCard({ data, referral = null, coordinates = null }: ResponseCardProps) {
  const [activeTab, setActiveTab] = useState<Tab>("diagnosis");
  const doctorState = useDoctorRecommendations({
    referral,
    coordinates,
    primaryDoctors: data.doctors,
  });
  const doctorsForDisplay: DisplayDoctor[] = referral
    ? doctorState.doctors.map(mapRecommendedDoctorToDisplay)
    : data.doctors.map(mapApiDoctorToDisplay);

  const tabs: { key: Tab; label: string; emoji: string }[] = [
    { key: "diagnosis", label: "Diagnosis", emoji: "🩺" },
    { key: "lifestyle", label: "Lifestyle", emoji: "🥗" },
    { key: "doctors", label: "Doctors", emoji: "👨‍⚕️" },
    { key: "sources", label: "Sources", emoji: "📚" },
  ];

  return (
    <div className="w-full overflow-hidden rounded-2xl border border-[#ebebef] bg-white shadow-sm">
      <div className="flex border-b border-[#ebebef] bg-[#fafafa]">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setActiveTab(tab.key)}
            className={`flex-1 py-3 text-sm font-semibold transition ${
              activeTab === tab.key
                ? "border-b-2 border-[#6f4ef2] bg-white text-[#6f4ef2]"
                : "text-[#6b6b76] hover:text-[#111]"
            }`}
          >
            {tab.emoji} {tab.label}
          </button>
        ))}
      </div>

      <div className="p-5">
        {activeTab === "diagnosis" && (
          <div className="space-y-4">
            {data.answer && (
              <div>
                <p className="mb-2 text-xs font-bold uppercase tracking-widest text-[#7b7b88]">
                  Clinical response
                </p>
                <div className="text-sm leading-relaxed whitespace-pre-line text-[#1a1a1a]">
                  {data.answer}
                </div>
              </div>
            )}

            {data.suspected_conditions.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-bold uppercase tracking-widest text-[#6f4ef2]">
                  Suspected conditions
                </p>
                <div className="flex flex-wrap gap-2">
                  {data.suspected_conditions.map((c, i) => (
                    <span
                      key={i}
                      className="rounded-full border border-[#e4d9ff] bg-[#f4eeff] px-3 py-1 text-sm font-medium text-[#5d42d4]"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {data.symptoms.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-bold uppercase tracking-widest text-[#7b7b88]">
                  Detected symptoms
                </p>
                <div className="flex flex-wrap gap-2">
                  {data.symptoms.map((s, i) => (
                    <span
                      key={i}
                      className="rounded-full border border-[#ebebef] bg-[#fafafa] px-3 py-1 text-sm text-[#3a3a42]"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <p className="border-t border-[#ebebef] pt-3 text-xs text-[#8f8f95]">
              For educational purposes only. Not a substitute for professional medical advice.
            </p>
          </div>
        )}

        {activeTab === "lifestyle" && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <RecommendationSection title="Foods to Eat" emoji="✅" tone="green" items={data.recommendations.foods_to_eat} />
            <RecommendationSection title="Foods to Avoid" emoji="❌" tone="rose" items={data.recommendations.foods_to_avoid} />
            <RecommendationSection title="Drinks to Have" emoji="💧" tone="sky" items={data.recommendations.drinks_to_have} />
            <RecommendationSection title="Drinks to Avoid" emoji="🚫" tone="amber" items={data.recommendations.drinks_to_avoid} />
            <RecommendationSection title="Exercise — Do" emoji="🏃" tone="violet" items={data.recommendations.exercises_recommended} />
            <RecommendationSection title="Exercise — Avoid" emoji="🛑" tone="rose" items={data.recommendations.exercises_to_avoid} />
            {data.recommendations.rest_recommendation && (
              <div className="rounded-xl border border-[#ebebef] bg-[#fafafa] p-4 md:col-span-2">
                <p className="mb-1 text-xs font-bold uppercase tracking-widest text-[#6f4ef2]">
                  Rest recommendation
                </p>
                <p className="text-sm text-[#1a1a1a]">{data.recommendations.rest_recommendation}</p>
              </div>
            )}
          </div>
        )}

        {activeTab === "doctors" && (
          <div className="space-y-3">
            {referral && doctorState.loading ? (
              <div className="space-y-3">
                <div className="animate-pulse rounded-xl border border-[#ebebef] bg-[#fafafa] p-4">
                  <div className="h-4 w-40 rounded bg-[#ececf2]" />
                  <div className="mt-3 h-3 w-64 rounded bg-[#ececf2]" />
                </div>
              </div>
            ) : doctorsForDisplay.length === 0 ? (
              <p className="py-6 text-center text-sm text-[#8f8f95]">
                {referral && doctorState.error
                  ? doctorState.error
                  : "No doctor results available for the detected conditions."}
              </p>
            ) : (
              <>
                {referral && coordinates && (
                  <p className="text-xs text-[#8f8f95]">
                    Nearby Google Maps results are shown first using your current location, with additional registry matches listed after them when available.
                  </p>
                )}
                {doctorsForDisplay.map((doc, i) => <DoctorCard key={i} doctor={doc} />)}
              </>
            )}
          </div>
        )}

        {activeTab === "sources" && (
          <div className="space-y-2">
            {data.sources.length === 0 ? (
              <p className="py-6 text-center text-sm text-[#8f8f95]">No sources available.</p>
            ) : (
              data.sources.map((s, i) => <SourceBadge key={i} source={s} />)
            )}
          </div>
        )}
      </div>
    </div>
  );
}

type Tone = "green" | "rose" | "sky" | "amber" | "violet";

const toneClass: Record<Tone, string> = {
  green:  "border-emerald-200 bg-emerald-50 text-emerald-900",
  rose:   "border-rose-200 bg-rose-50 text-rose-900",
  sky:    "border-sky-200 bg-sky-50 text-sky-900",
  amber:  "border-amber-200 bg-amber-50 text-amber-900",
  violet: "border-violet-200 bg-violet-50 text-violet-900",
};

function RecommendationSection({
  title,
  emoji,
  tone,
  items,
}: {
  title: string;
  emoji: string;
  tone: Tone;
  items: string[];
}) {
  if (!items?.length) return null;
  return (
    <div className={`rounded-xl border p-4 ${toneClass[tone]}`}>
      <p className="mb-2 text-xs font-bold uppercase tracking-widest opacity-80">
        {emoji} {title}
      </p>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2 text-sm">
            <span className="mt-1 text-xs opacity-60">•</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DoctorCard({ doctor }: { doctor: DisplayDoctor }) {
  return (
    <div className="rounded-xl border border-[#ebebef] bg-[#fafafa] p-4 transition hover:border-[#d9c5ff]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-semibold text-[#111]">{doctor.name}</p>
          <p className="text-sm text-[#6f4ef2]">{doctor.specialty}</p>
          <p className="mt-1 text-xs text-[#6b6b76]">{doctor.address}</p>
          {doctor.phone ? <p className="text-xs text-[#6b6b76]">{doctor.phone}</p> : null}
          {doctor.rating !== null ? (
            <p className="mt-1 text-xs text-[#6b6b76]">
              ⭐ {doctor.rating.toFixed(1)} ({doctor.reviewCount})
            </p>
          ) : null}
          {doctor.mapsUrl ? (
            <a
              href={doctor.mapsUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-block text-xs font-medium text-[#6f4ef2] hover:underline"
            >
              View on Maps
            </a>
          ) : null}
        </div>
        <div className="shrink-0 text-right">
          <span className={`rounded-full border px-2 py-1 text-xs font-medium ${doctor.badgeTone}`}>
            {doctor.badgeLabel}
          </span>
          <p className="mt-1 text-xs text-[#8f8f95]">{doctor.sourceText}</p>
          {doctor.reference ? (
            <p className="mt-1 text-[11px] text-[#b09adf]">{doctor.reference}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function SourceBadge({ source }: { source: Source }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-[#ebebef] bg-[#fafafa] p-3">
      <span className="text-lg">📘</span>
      <div>
        <p className="text-sm font-medium text-[#111]">{source.book}</p>
        {source.section && <p className="text-xs text-[#6b6b76]">{source.section}</p>}
      </div>
    </div>
  );
}
