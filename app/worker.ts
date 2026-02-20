import { env, pipeline } from "@xenova/transformers/dist/transformers.js";
import type {
  AutomaticSpeechRecognitionConfig,
  AutomaticSpeechRecognitionPipeline,
} from "@xenova/transformers";

type WhisperLanguage =
  | "english"
  | "turkish"
  | "spanish"
  | "french"
  | "german"
  | "italian"
  | "portuguese"
  | "russian"
  | "arabic"
  | "hindi"
  | "japanese"
  | "korean";

type ProgressPhase = "download" | "transcribing";

type TranscriptSegment = {
  text: string;
  start: number;
  end: number;
};

type WorkerRequest =
  | { type: "load" }
  | {
      type: "transcribe";
      requestId: number;
      audio: Float32Array;
      language?: "auto" | WhisperLanguage;
    };

type WorkerStatus = "loading" | "ready" | "transcribing" | "error";

type WorkerResponse =
  | { type: "status"; status: WorkerStatus; requestId?: number; detail?: string }
  | {
      type: "progress";
      phase: ProgressPhase;
      progress: number;
      requestId?: number;
      processedChunks?: number;
      totalChunks?: number;
      loaded?: number;
      total?: number;
      file?: string;
    }
  | { type: "partial"; text: string; requestId: number }
  | { type: "segments"; requestId: number; text: string; segments: TranscriptSegment[] }
  | { type: "result"; text: string; requestId: number }
  | { type: "error"; error: string; requestId?: number };

type DecodedChunk = {
  tokens: number[];
  stride: number[];
  token_timestamps?: number[];
};

const SAMPLE_RATE = 16_000;
const CHUNK_LENGTH_SECONDS = 30;
const STRIDE_LENGTH_SECONDS = 5;
const CHUNK_WINDOW_SIZE = SAMPLE_RATE * CHUNK_LENGTH_SECONDS;
const STRIDE_SIZE = SAMPLE_RATE * STRIDE_LENGTH_SECONDS;
const CHUNK_JUMP_SIZE = CHUNK_WINDOW_SIZE - 2 * STRIDE_SIZE;
const ASR_MODEL_ID = "Xenova/whisper-small";

env.allowLocalModels = false;
env.useBrowserCache = true;

type WorkerScope = {
  postMessage: (message: WorkerResponse) => void;
  addEventListener: (
    type: "message",
    listener: (event: MessageEvent<WorkerRequest>) => void | Promise<void>,
  ) => void;
};

const workerScope = self as unknown as WorkerScope;

let transcriberPromise: Promise<AutomaticSpeechRecognitionPipeline> | null = null;

function postToMain(message: WorkerResponse) {
  workerScope.postMessage(message);
}

function normalizeProgress(rawProgress: unknown): number {
  if (typeof rawProgress !== "number" || Number.isNaN(rawProgress)) {
    return 0;
  }
  const normalized = rawProgress <= 1 ? rawProgress * 100 : rawProgress;
  return Math.max(0, Math.min(100, normalized));
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "Unknown worker error.";
}

function estimateChunkCount(audioLength: number): number {
  if (audioLength <= CHUNK_WINDOW_SIZE || CHUNK_JUMP_SIZE <= 0) {
    return 1;
  }
  return Math.ceil((audioLength - CHUNK_WINDOW_SIZE) / CHUNK_JUMP_SIZE) + 1;
}

function toSafeTimestamp(value: unknown, fallback: number): number {
  if (typeof value !== "number" || Number.isNaN(value) || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(0, value);
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function collapseConsecutiveRepeatedNgrams(text: string): string {
  const normalized = normalizeWhitespace(text);
  if (!normalized) return "";

  const words = normalized.split(" ");
  if (words.length < 6) {
    return normalized;
  }

  for (let gramSize = Math.min(20, Math.floor(words.length / 2)); gramSize >= 3; gramSize -= 1) {
    let index = gramSize * 2;
    while (index <= words.length) {
      let repeated = true;
      for (let offset = 0; offset < gramSize; offset += 1) {
        if (words[index - gramSize * 2 + offset] !== words[index - gramSize + offset]) {
          repeated = false;
          break;
        }
      }

      if (repeated) {
        words.splice(index - gramSize, gramSize);
      } else {
        index += 1;
      }
    }
  }

  return words.join(" ").trim();
}

function normalizeForCompare(text: string): string {
  return collapseConsecutiveRepeatedNgrams(text).toLowerCase();
}

function dedupeSegments(segments: TranscriptSegment[]): TranscriptSegment[] {
  const sorted = [...segments].sort((a, b) => a.start - b.start || a.end - b.end);
  const deduped: TranscriptSegment[] = [];

  for (const segment of sorted) {
    const cleanedText = collapseConsecutiveRepeatedNgrams(segment.text);
    if (!cleanedText) continue;

    const next: TranscriptSegment = {
      text: cleanedText,
      start: segment.start,
      end: Math.max(segment.end, segment.start),
    };

    const prev = deduped[deduped.length - 1];
    if (!prev) {
      deduped.push(next);
      continue;
    }

    const prevNormalized = normalizeForCompare(prev.text);
    const nextNormalized = normalizeForCompare(next.text);
    const overlaps = next.start <= prev.end + 0.35;
    const sameOrContained =
      prevNormalized === nextNormalized ||
      prevNormalized.includes(nextNormalized) ||
      nextNormalized.includes(prevNormalized);

    if (sameOrContained && overlaps) {
      if (nextNormalized.length > prevNormalized.length) {
        prev.text = next.text;
      }
      prev.end = Math.max(prev.end, next.end);
      continue;
    }

    deduped.push(next);
  }

  return deduped;
}

function decodeChunksToText(
  transcriber: AutomaticSpeechRecognitionPipeline,
  chunks: DecodedChunk[],
): string | null {
  const validChunks = chunks.filter(
    (chunk) =>
      Array.isArray(chunk.tokens) &&
      chunk.tokens.length > 0 &&
      chunk.tokens.every((token) => Number.isInteger(token)) &&
      Array.isArray(chunk.stride),
  );

  const tokenizer = transcriber.tokenizer as {
    _decode_asr?: (
      sequences: DecodedChunk[],
      options?: {
        return_timestamps?: boolean;
        force_full_sequences?: boolean;
        time_precision?: number;
      },
    ) => unknown[];
  };

  if (typeof tokenizer._decode_asr !== "function" || validChunks.length === 0) {
    return null;
  }

  const featureExtractor = (transcriber.processor as { feature_extractor?: { config?: unknown } })
    .feature_extractor;
  const config = featureExtractor?.config as { chunk_length?: number } | undefined;
  const modelConfig = transcriber.model.config as { max_source_positions?: number };

  const timePrecision =
    typeof config?.chunk_length === "number" &&
    typeof modelConfig.max_source_positions === "number" &&
    modelConfig.max_source_positions > 0
      ? config.chunk_length / modelConfig.max_source_positions
      : undefined;

  const decoded = tokenizer._decode_asr(validChunks, {
    return_timestamps: false,
    force_full_sequences: false,
    time_precision: timePrecision,
  });

  const first = Array.isArray(decoded) ? decoded[0] : null;
  if (typeof first === "string") {
    return first.trim();
  }
  if (first && typeof first === "object" && "text" in first) {
    const text = (first as { text?: unknown }).text;
    if (typeof text === "string") {
      return text.trim();
    }
  }
  return null;
}

async function getTranscriber(): Promise<AutomaticSpeechRecognitionPipeline> {
  if (!transcriberPromise) {
    postToMain({ type: "status", status: "loading", detail: "Initializing model..." });
    transcriberPromise = pipeline("automatic-speech-recognition", ASR_MODEL_ID, {
      quantized: true,
      progress_callback: (progressInfo: {
        status?: string;
        progress?: number;
        loaded?: number;
        total?: number;
        file?: string;
      }) => {
        postToMain({
          type: "status",
          status: "loading",
          detail: progressInfo.status ?? "Loading model...",
        });
        postToMain({
          type: "progress",
          phase: "download",
          progress: normalizeProgress(progressInfo.progress),
          loaded: progressInfo.loaded,
          total: progressInfo.total,
          file: progressInfo.file,
        });
      },
    }) as Promise<AutomaticSpeechRecognitionPipeline>;
  }

  try {
    const transcriber = await transcriberPromise;
    postToMain({ type: "status", status: "ready", detail: "Model ready." });
    return transcriber;
  } catch (error) {
    transcriberPromise = null;
    const message = errorMessage(error);
    postToMain({ type: "status", status: "error", detail: message });
    postToMain({ type: "error", error: message });
    throw error;
  }
}

workerScope.addEventListener("message", async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  if (message.type === "load") {
    try {
      await getTranscriber();
    } catch {
      // Errors are already posted from getTranscriber().
    }
    return;
  }

  if (message.type !== "transcribe") {
    return;
  }

  const { requestId, audio } = message;
  const languageHint = message.language;

  try {
    const transcriber = await getTranscriber();
    postToMain({ type: "status", status: "transcribing", requestId });

    const chunks: DecodedChunk[] = [];
    let latestPartial = "";
    const totalChunks = estimateChunkCount(audio.length);
    let processedChunks = 0;

    postToMain({
      type: "progress",
      phase: "transcribing",
      progress: 0,
      requestId,
      processedChunks,
      totalChunks,
    });

    const transcriptionOptions: AutomaticSpeechRecognitionConfig = {
      return_timestamps: true,
      chunk_length_s: CHUNK_LENGTH_SECONDS,
      stride_length_s: STRIDE_LENGTH_SECONDS,
      task: "transcribe",
      temperature: 0,
      num_beams: 5,
      no_repeat_ngram_size: 3,
      repetition_penalty: 1.3,
      chunk_callback: (chunk: {
        tokens?: number[];
        stride?: number[];
        token_timestamps?: number[];
      }) => {
        processedChunks = Math.min(totalChunks, processedChunks + 1);
        postToMain({
          type: "progress",
          phase: "transcribing",
          progress: normalizeProgress((processedChunks / totalChunks) * 100),
          requestId,
          processedChunks,
          totalChunks,
        });

        if (
          !Array.isArray(chunk.tokens) ||
          chunk.tokens.length === 0 ||
          !chunk.tokens.every((token) => Number.isInteger(token)) ||
          !Array.isArray(chunk.stride)
        ) {
          return;
        }

        chunks.push({
          tokens: chunk.tokens,
          stride: chunk.stride,
          token_timestamps: chunk.token_timestamps,
        });

        try {
          const decoded = decodeChunksToText(transcriber, chunks);
          if (!decoded) {
            return;
          }

          const cleanedPartial = normalizeWhitespace(decoded);
          if (cleanedPartial && cleanedPartial !== latestPartial) {
            latestPartial = cleanedPartial;
            postToMain({ type: "partial", text: cleanedPartial, requestId });
          }
        } catch {
          // Partial decoding is best-effort. Ignore transient chunk decode errors.
        }
      },
    };

    if (languageHint && languageHint !== "auto") {
      transcriptionOptions.language = languageHint;
    }

    const output = await transcriber(audio, transcriptionOptions);

    postToMain({
      type: "progress",
      phase: "transcribing",
      progress: 100,
      requestId,
      processedChunks: totalChunks,
      totalChunks,
    });

    const normalizedOutput = Array.isArray(output) ? output[0] : output;
    const finalText = collapseConsecutiveRepeatedNgrams((normalizedOutput?.text ?? "").trim());
    const rawChunks = Array.isArray(normalizedOutput?.chunks) ? normalizedOutput.chunks : [];

    const mappedSegments: TranscriptSegment[] = rawChunks
      .map((chunk) => {
        const text = typeof chunk.text === "string" ? chunk.text.trim() : "";
        const rawStart = Array.isArray(chunk.timestamp) ? chunk.timestamp[0] : 0;
        const start = toSafeTimestamp(rawStart, 0);
        const rawEnd = Array.isArray(chunk.timestamp) ? chunk.timestamp[1] : start;
        const end = toSafeTimestamp(rawEnd, start);
        return { text, start, end };
      })
      .filter((segment) => segment.text.length > 0);

    let segments = dedupeSegments(mappedSegments);
    if (segments.length === 0 && finalText) {
      segments = [{ text: finalText, start: 0, end: 0 }];
    }

    const finalTextFromSegments = collapseConsecutiveRepeatedNgrams(
      segments.map((segment) => segment.text).join(" "),
    );
    const canonicalText = finalTextFromSegments || finalText;

    postToMain({ type: "segments", requestId, text: canonicalText, segments });
    postToMain({ type: "result", text: canonicalText, requestId });
    postToMain({ type: "status", status: "ready", requestId });
  } catch (error) {
    const messageText = errorMessage(error);
    postToMain({ type: "status", status: "error", requestId, detail: messageText });
    postToMain({ type: "error", error: messageText, requestId });
  }
});

export {};
