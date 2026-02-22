import { env, pipeline } from "@huggingface/transformers";
import type { AutomaticSpeechRecognitionPipeline } from "@huggingface/transformers";

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
  | { type: "status"; status: WorkerStatus; requestId?: number; detail?: string; device?: string }
  | {
      type: "progress";
      phase: ProgressPhase;
      progress: number;
      requestId?: number;
      processedChunks?: number;
      totalChunks?: number;
      currentSlice?: number;
      totalSlices?: number;
      loaded?: number;
      total?: number;
      file?: string;
    }
  | { type: "partial"; text: string; requestId: number }
  | { type: "segments"; requestId: number; text: string; segments: TranscriptSegment[] }
  | { type: "result"; text: string; requestId: number }
  | { type: "error"; error: string; requestId?: number };

// ─── Constants ───────────────────────────────────────────────────────────────

const ASR_MODEL_ID = "Xenova/whisper-small";
const CHUNK_LENGTH_S = 30;
const STRIDE_LENGTH_S = 5;
const SAMPLE_RATE = 16_000;
const CHUNK_JUMP_SAMPLES = (CHUNK_LENGTH_S - 2 * STRIDE_LENGTH_S) * SAMPLE_RATE;

// ─── Worker environment ───────────────────────────────────────────────────────

env.allowLocalModels = false;
env.useBrowserCache = true;
// Suppress the ORT "Some nodes were not assigned to the preferred EP" warning.
// ORT deliberately routes shape-related ops to CPU even on WebGPU — this is
// intentional and has no negative impact; the warning is purely informational.
// Setting logSeverityLevel to 3 (Error) silences Warning-level ORT messages.
(env.backends as Record<string, Record<string, unknown>>).onnx ??= {};
(env.backends as Record<string, Record<string, unknown>>).onnx.logSeverityLevel = 3;

type WorkerScope = {
  postMessage: (message: WorkerResponse) => void;
  addEventListener: (
    type: "message",
    listener: (event: MessageEvent<WorkerRequest>) => void | Promise<void>,
  ) => void;
};

const workerScope = self as unknown as WorkerScope;

// ─── Pipeline singleton ───────────────────────────────────────────────────────

let transcriberPromise: Promise<AutomaticSpeechRecognitionPipeline> | null = null;
/** Tracks which device the active pipeline was initialised on. */
let activeDevice: "webgpu" | "wasm" | null = null;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function postToMain(message: WorkerResponse): void {
  workerScope.postMessage(message);
}

function normalizeProgress(rawProgress: unknown): number {
  if (typeof rawProgress !== "number" || Number.isNaN(rawProgress)) return 0;
  const normalized = rawProgress <= 1 ? rawProgress * 100 : rawProgress;
  return Math.max(0, Math.min(100, normalized));
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  return "Unknown worker error.";
}

function estimateChunkCount(audioLength: number): number {
  const windowSamples = CHUNK_LENGTH_S * SAMPLE_RATE;
  if (audioLength <= windowSamples || CHUNK_JUMP_SAMPLES <= 0) return 1;
  return Math.ceil((audioLength - windowSamples) / CHUNK_JUMP_SAMPLES) + 1;
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
  if (words.length < 6) return normalized;

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

    const prevNorm = normalizeForCompare(prev.text);
    const nextNorm = normalizeForCompare(next.text);
    const overlaps = next.start <= prev.end + 0.35;
    const sameOrContained =
      prevNorm === nextNorm || prevNorm.includes(nextNorm) || nextNorm.includes(prevNorm);

    if (sameOrContained && overlaps) {
      if (nextNorm.length > prevNorm.length) prev.text = next.text;
      prev.end = Math.max(prev.end, next.end);
      continue;
    }

    deduped.push(next);
  }

  return deduped;
}

// ─── Pipeline initialisation (WebGPU → WASM fallback) ─────────────────────────

type DownloadProgressInfo = {
  status?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  file?: string;
};

/**
 * progress_callback wired to the UI Progress Bar.
 * Called by @huggingface/transformers while model files are being downloaded.
 * Bind this to any UI progress element via the "progress" + "download" messages
 * received on the main thread.
 */
function onDownloadProgress(info: DownloadProgressInfo): void {
  // "ready" fires once ALL model shards are downloaded and the weights are
  // being loaded into GPU memory. WebGPU then begins shader compilation —
  // a silent phase that can take 1–3 minutes with no further progress events.
  // Post a distinct "compiling" detail so the UI can show a clear message.
  if (info.status === "ready") {
    postToMain({ type: "status", status: "loading", detail: "compiling" });
    return;
  }

  postToMain({ type: "status", status: "loading", detail: info.status ?? "Loading model…" });
  postToMain({
    type: "progress",
    phase: "download",
    progress: normalizeProgress(info.progress),
    loaded: info.loaded,
    total: info.total,
    file: info.file,
  });
}

async function buildPipeline(
  device: "webgpu" | "wasm",
): Promise<AutomaticSpeechRecognitionPipeline> {
  postToMain({
    type: "status",
    status: "loading",
    detail: `Initialising model on ${device.toUpperCase()}…`,
  });

  // WebGPU: fp16 encoder + q4 decoder — fp16 is native to GPU shaders and
  // ~2× faster than fp32 with negligible accuracy difference on whisper-small.
  // WASM fallback: q8 quantisation for acceptable CPU-only inference speed.
  const options =
    device === "webgpu"
      ? {
          dtype: {
            encoder_model: "fp16" as const,
            decoder_model_merged: "q4" as const,
          },
          device: "webgpu" as const,
          progress_callback: onDownloadProgress,
          session_options: { logSeverityLevel: 3 as const },
        }
      : {
          dtype: "q8" as const,
          device: "wasm" as const,
          progress_callback: onDownloadProgress,
          session_options: { logSeverityLevel: 3 as const },
        };

  // Cast through `unknown` to avoid the overly-complex union type that
  // @huggingface/transformers v3 pipeline overloads produce for TS.
  return pipeline(
    "automatic-speech-recognition",
    ASR_MODEL_ID,
    options,
  ) as unknown as Promise<AutomaticSpeechRecognitionPipeline>;
}

async function getTranscriber(): Promise<AutomaticSpeechRecognitionPipeline> {
  if (!transcriberPromise) {
    postToMain({ type: "status", status: "loading", detail: "Initialising model…" });

    transcriberPromise = (async () => {
      // ── Try WebGPU first; gracefully fall back to WASM when unsupported ──
      try {
        const transcriber = await buildPipeline("webgpu");
        activeDevice = "webgpu";
        return transcriber;
      } catch (webGpuError) {
        console.warn(
          "[worker] WebGPU unavailable — falling back to WASM:",
          errorMessage(webGpuError),
        );
        postToMain({
          type: "status",
          status: "loading",
          detail: "WebGPU not available — falling back to WASM…",
        });
        const transcriber = await buildPipeline("wasm");
        activeDevice = "wasm";
        return transcriber;
      }
    })();
  }

  try {
    const transcriber = await transcriberPromise;
    postToMain({
      type: "status",
      status: "ready",
      detail: `Model ready (${activeDevice?.toUpperCase() ?? "unknown"}).`,
      device: activeDevice ?? undefined,
    });
    return transcriber;
  } catch (error) {
    transcriberPromise = null;
    activeDevice = null;
    const msg = errorMessage(error);
    postToMain({ type: "status", status: "error", detail: msg });
    postToMain({ type: "error", error: msg });
    throw error;
  }
}

// ─── Message handler ──────────────────────────────────────────────────────────

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

  if (message.type !== "transcribe") return;

  const { requestId, audio } = message;
  const languageHint = message.language;

  try {
    const transcriber = await getTranscriber();
    postToMain({ type: "status", status: "transcribing", requestId, device: activeDevice ?? undefined });

    // ── Own 30-second windowing with overlap ─────────────────────────────────
    //
    // @huggingface/transformers v3 does NOT support `chunk_callback`.
    // The pipeline's internal chunking processes ALL windows in a blocking
    // loop with no progress events → the UI appears frozen for minutes.
    //
    // Instead we create overlapping 30-second windows ourselves, feed each
    // one to the pipeline as a single-chunk call (chunk_length_s = 0), and
    // emit progress after every `await transcriber()` return.
    //
    // Using `audio.slice()` (not `subarray()`) is critical: subarray shares
    // the original ArrayBuffer; the ONNX WebGPU backend may read the entire
    // buffer via `Float32Array(audio.buffer)` → OOM on large files.

    const WINDOW_SAMPLES = CHUNK_LENGTH_S * SAMPLE_RATE;   // 30 s = 480 000
    const STRIDE_SAMPLES = STRIDE_LENGTH_S * SAMPLE_RATE;  // 5 s  =  80 000
    const JUMP_SAMPLES   = CHUNK_JUMP_SAMPLES;             // 20 s = 320 000

    // Minimum window length the model can reliably produce tokens for.
    // Shorter clips are zero-padded to this length to prevent the tokenizer
    // from throwing "token_ids must be a non-empty array of integers".
    const MIN_WINDOW_SAMPLES = 1 * SAMPLE_RATE; // 1 second

    /** Zero-pad a short buffer to at least `minLen` samples. */
    function padToMinLength(buf: Float32Array, minLen: number): Float32Array {
      if (buf.length >= minLen) return buf;
      const padded = new Float32Array(minLen); // zeros by default
      padded.set(buf);
      return padded;
    }

    // Build the list of overlapping windows
    type Window = { data: Float32Array; offsetS: number };
    const windows: Window[] = [];
    let offset = 0;
    while (offset < audio.length) {
      const end = Math.min(offset + WINDOW_SAMPLES, audio.length);
      const slice = audio.slice(offset, end);
      // Pad short tail windows with silence instead of skipping them.
      const windowData = padToMinLength(slice, MIN_WINDOW_SAMPLES);
      windows.push({ data: windowData, offsetS: offset / SAMPLE_RATE });
      if (end >= audio.length) break;
      offset += JUMP_SAMPLES;
    }

    // Safety net: if audio was empty, push the whole thing.
    if (windows.length === 0 && audio.length > 0) {
      windows.push({ data: padToMinLength(audio.slice(0), MIN_WINDOW_SAMPLES), offsetS: 0 });
    }

    const totalWindows = windows.length;

    postToMain({
      type: "progress",
      phase: "transcribing",
      progress: 0,
      requestId,
      processedChunks: 0,
      totalChunks: totalWindows,
      currentSlice: 1,
      totalSlices: 1,
    });

    // ── Accumulators ─────────────────────────────────────────────────────────
    const accumulatedTexts: string[] = [];
    const allSegments: TranscriptSegment[] = [];

    // ── Process each window sequentially ─────────────────────────────────────
    for (let i = 0; i < totalWindows; i++) {
      const win = windows[i];

      try {
        const result = await transcriber(win.data, {
          return_timestamps: true,
          // chunk_length_s = 30 → the standard Whisper window size.
          // Since every window is already ≤ 30 s the pipeline will treat it
          // as a single internal chunk with no further sub-windowing.
          chunk_length_s: CHUNK_LENGTH_S,
          stride_length_s: 0,
          // Greedy decoding – one forward pass per chunk, no beam-search overhead.
          temperature: 0,
          num_beams: 1,
          language: languageHint && languageHint !== "auto" ? languageHint : undefined,
        } as Record<string, unknown>);

        const normalized = Array.isArray(result) ? result[0] : result;

        // ── Collect segments and offset timestamps ──────────────────────────
        const rawChunks = Array.isArray(normalized?.chunks) ? normalized.chunks : [];
        for (const chunk of rawChunks) {
          const text = typeof chunk.text === "string" ? chunk.text.trim() : "";
          if (!text) continue;
          const ts = Array.isArray(chunk.timestamp) ? chunk.timestamp : [0, 0];
          const start = toSafeTimestamp(ts[0], 0) + win.offsetS;
          const end   = toSafeTimestamp(ts[1], toSafeTimestamp(ts[0], 0)) + win.offsetS;
          allSegments.push({ text, start, end });
        }

        // ── Live preview ────────────────────────────────────────────────────
        const windowText = normalizeWhitespace(normalized?.text ?? "");
        if (windowText) {
          accumulatedTexts.push(windowText);
          postToMain({ type: "partial", text: accumulatedTexts.join(" "), requestId });
        }
      } catch (windowError) {
        // Gracefully skip windows where the model produces zero tokens.
        // This typically happens on silent or very noisy segments where
        // Whisper's decoder emits an empty token_ids array.
        console.warn(
          `[worker] Window ${i + 1}/${totalWindows} failed — skipping:`,
          errorMessage(windowError),
        );
      }

      // ── Progress ──────────────────────────────────────────────────────────
      const processed = i + 1;
      postToMain({
        type: "progress",
        phase: "transcribing",
        progress: normalizeProgress((processed / totalWindows) * 100),
        requestId,
        processedChunks: processed,
        totalChunks: totalWindows,
        currentSlice: 1,
        totalSlices: 1,
      });
    }

    // ── Merge + deduplicate segments ─────────────────────────────────────────
    let segments = dedupeSegments(allSegments);

    const rawFullText = collapseConsecutiveRepeatedNgrams(
      accumulatedTexts.join(" "),
    );

    if (segments.length === 0 && rawFullText) {
      segments = [{ text: rawFullText, start: 0, end: 0 }];
    }

    const canonicalText =
      collapseConsecutiveRepeatedNgrams(segments.map((s) => s.text).join(" ")) ||
      rawFullText;

    postToMain({ type: "segments", requestId, text: canonicalText, segments });
    postToMain({ type: "result",   text: canonicalText, requestId });
    postToMain({ type: "status",   status: "ready", requestId });
  } catch (error) {
    const msg = errorMessage(error);
    postToMain({ type: "status", status: "error", requestId, detail: msg });
    postToMain({ type: "error",  error: msg, requestId });
  }
});

export {};

