import { NextResponse } from "next/server";

const LANGUAGE_CODES: Record<string, string> = {
    english: "en",
    turkish: "tr",
    spanish: "es",
    french: "fr",
    german: "de",
    italian: "it",
    portuguese: "pt",
    russian: "ru",
    arabic: "ar",
    hindi: "hi",
    japanese: "ja",
    korean: "ko",
};

export async function POST(req: Request) {
    try {
        const formData = await req.formData();
        const audioFile = formData.get("file") as File;
        const language = formData.get("language") as string;

        if (!audioFile) {
            return NextResponse.json({ error: "No audio file provided" }, { status: 400 });
        }

        if (!process.env.GROQ_API_KEY) {
            return NextResponse.json({ error: "GROQ_API_KEY is not configured on the server." }, { status: 500 });
        }

        const groqFormData = new FormData();
        groqFormData.append("file", audioFile);
        groqFormData.append("model", "whisper-large-v3");
        groqFormData.append("response_format", "verbose_json");

        // Only pass language to Groq if it's explicitly set and not "auto"
        if (language && language !== "auto" && LANGUAGE_CODES[language]) {
            groqFormData.append("language", LANGUAGE_CODES[language]);
        }

        const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
            method: "POST",
            headers: {
                Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
            },
            body: groqFormData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return NextResponse.json(
                { error: errorData.error?.message || "Failed to transcribe audio on Groq." },
                { status: response.status }
            );
        }

        const data = await response.json();

        return NextResponse.json({
            text: data.text,
            segments: data.segments?.map((s: { text: string; start: number; end: number }) => ({
                text: s.text,
                start: s.start,
                end: s.end
            })) || []
        });

    } catch (error) {
        console.error("Groq Transcription Error:", error);
        return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
    }
}
