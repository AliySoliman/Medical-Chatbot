import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const incomingFormData = await request.formData();
    const file = incomingFormData.get("file");

    if (!(file instanceof Blob)) {
      throw new Error("Missing audio file");
    }

    const mimeType = file.type || "audio/webm";
    const extension = mimeType.includes("ogg") ? "ogg" : "webm";

    const formData = new FormData();
    formData.append("file", file, `recording.${extension}`);
    formData.append("model", "whisper-large-v3-turbo");
    formData.append("response_format", "json");
    formData.append("language", "en");

    const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: formData,
    });

    const data = (await response.json()) as { text?: string };

    if (!response.ok) {
      throw new Error("Groq transcription request failed");
    }

    return NextResponse.json({ text: data.text ?? "" });
  } catch {
    return NextResponse.json({ error: "Transcription failed" }, { status: 500 });
  }
}
