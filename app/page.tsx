"use client";

import {
  AlertCircle,
  Check,
  ChevronDown,
  Clock3,
  Copy,
  Download,
  FileJson,
  LoaderCircle,
  ShieldCheck,
  Square,
  Type,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { UploadDropzone } from "@/components/upload-dropzone";

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

type LanguageOption = { value: "auto" | WhisperLanguage; label: string };
type ProgressPhase = "download" | "transcribing";
type CopyState = "idle" | "success" | "error";
type FontMode = "sans" | "mono";
type SmartExportAction = "copy_text_only" | "copy_with_timestamps" | "export_json";

type WorkerStatus = "loading" | "ready" | "transcribing" | "error";
type TranscriptionStatus = "idle" | "loading" | "decoding" | "transcribing" | "ready" | "error";

type WorkerRequest =
  | { type: "load" }
  | {
      type: "transcribe";
      requestId: number;
      audio: Float32Array;
      language?: "auto" | WhisperLanguage;
    };

type TranscriptSegment = {
  text: string;
  start: number;
  end: number;
};

type TranscriptExportJson = {
  version: 1;
  createdAt: string;
  fileName: string | null;
  model: "Xenova/whisper-small";
  language: "auto" | WhisperLanguage;
  text: string;
  segments: TranscriptSegment[];
};

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

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: "auto", label: "Auto detect" },
  { value: "english", label: "English" },
  { value: "turkish", label: "Turkish" },
  { value: "spanish", label: "Spanish" },
  { value: "french", label: "French" },
  { value: "german", label: "German" },
  { value: "italian", label: "Italian" },
  { value: "portuguese", label: "Portuguese" },
  { value: "russian", label: "Russian" },
  { value: "arabic", label: "Arabic" },
  { value: "hindi", label: "Hindi" },
  { value: "japanese", label: "Japanese" },
  { value: "korean", label: "Korean" },
];

function statusLabel(status: TranscriptionStatus): string {
  if (status === "idle") return "Idle";
  if (status === "loading") return "Loading model";
  if (status === "decoding") return "Decoding audio";
  if (status === "transcribing") return "Transcribing";
  if (status === "ready") return "Ready";
  return "Error";
}

function clampProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function downmixToMono(audioBuffer: AudioBuffer): Float32Array {
  const { numberOfChannels, length } = audioBuffer;
  if (numberOfChannels === 1) {
    const mono = new Float32Array(length);
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }

  const mono = new Float32Array(length);
  for (let channel = 0; channel < numberOfChannels; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let i = 0; i < length; i += 1) {
      mono[i] += channelData[i];
    }
  }

  for (let i = 0; i < length; i += 1) {
    mono[i] /= numberOfChannels;
  }

  return mono;
}

function resampleMonoAudio(
  input: Float32Array,
  inputSampleRate: number,
  outputSampleRate: number,
): Float32Array {
  if (inputSampleRate === outputSampleRate) {
    return input;
  }

  const ratio = outputSampleRate / inputSampleRate;
  const outputLength = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const sourceIndex = i / ratio;
    const lower = Math.floor(sourceIndex);
    const upper = Math.min(lower + 1, input.length - 1);
    const weight = sourceIndex - lower;
    output[i] = input[lower] * (1 - weight) + input[upper] * weight;
  }

  return output;
}

async function decodeAudioFile(file: File): Promise<Float32Array> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextClass = window.AudioContext;
  if (!AudioContextClass) {
    throw new Error("Web Audio API is not supported in this browser.");
  }

  const audioContext = new AudioContextClass();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const mono = downmixToMono(audioBuffer);
    return resampleMonoAudio(mono, audioBuffer.sampleRate, 16_000);
  } finally {
    await audioContext.close();
  }
}

function timestampForFilename(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return `${year}-${month}-${day}-${hours}-${minutes}-${seconds}`;
}

function formatSegmentTimestamp(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const secs = safeSeconds % 60;

  const paddedMinutes = String(minutes).padStart(2, "0");
  const paddedSeconds = String(secs).padStart(2, "0");

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${paddedMinutes}:${paddedSeconds}`;
  }
  return `${paddedMinutes}:${paddedSeconds}`;
}

export default function Home() {
  const workerRef = useRef<Worker | null>(null);
  const activeRequestIdRef = useRef(0);
  const copyResetTimeoutRef = useRef<number | null>(null);
  const outputTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const transcribeStartedAtRef = useRef<number | null>(null);

  const [status, setStatus] = useState<TranscriptionStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [progressPhase, setProgressPhase] = useState<ProgressPhase | null>(null);
  const [processedChunks, setProcessedChunks] = useState<number | null>(null);
  const [totalChunks, setTotalChunks] = useState<number | null>(null);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [output, setOutput] = useState("");
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeFileName, setActiveFileName] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<"auto" | WhisperLanguage>("auto");
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [fontMode, setFontMode] = useState<FontMode>("sans");
  const [isCancelling, setIsCancelling] = useState(false);
  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const [downloadedBytes, setDownloadedBytes] = useState<number | null>(null);
  const [totalBytes, setTotalBytes] = useState<number | null>(null);

  const clearProgressState = useCallback(() => {
    setProgress(0);
    setProgressPhase(null);
    setProcessedChunks(null);
    setTotalChunks(null);
    setEtaSeconds(null);
    setDownloadedBytes(null);
    setTotalBytes(null);
    transcribeStartedAtRef.current = null;
  }, []);

  const handleWorkerMessage = useCallback(
    (message: WorkerResponse) => {
      if (message.type === "status") {
        if (
          typeof message.requestId === "number" &&
          message.requestId !== activeRequestIdRef.current
        ) {
          return;
        }

        if (message.status === "loading") {
          setStatus("loading");
        } else if (message.status === "transcribing") {
          setStatus("transcribing");
        } else if (message.status === "ready") {
          setStatus("ready");
          setProgressPhase(null);
          setProcessedChunks(null);
          setTotalChunks(null);
          setEtaSeconds(null);
          transcribeStartedAtRef.current = null;
        } else if (message.status === "error") {
          setStatus("error");
          clearProgressState();
        }

        if (message.detail && message.status === "error") {
          setError(message.detail);
        }
        return;
      }

      if (message.type === "progress") {
        if (
          message.phase === "transcribing" &&
          typeof message.requestId === "number" &&
          message.requestId !== activeRequestIdRef.current
        ) {
          return;
        }

        setProgressPhase(message.phase);
        setProgress(clampProgress(message.progress));

        if (message.phase === "download") {
          setStatus("loading");
          setProcessedChunks(null);
          setTotalChunks(null);
          setEtaSeconds(null);
          transcribeStartedAtRef.current = null;
          if (typeof message.loaded === "number") setDownloadedBytes(message.loaded);
          if (typeof message.total === "number" && message.total > 0) setTotalBytes(message.total);
          return;
        }

        setStatus("transcribing");
        setProcessedChunks(message.processedChunks ?? null);
        setTotalChunks(message.totalChunks ?? null);

        if (transcribeStartedAtRef.current === null) {
          transcribeStartedAtRef.current = Date.now();
        }

        const processed = message.processedChunks;
        const total = message.totalChunks;
        if (
          typeof processed === "number" &&
          typeof total === "number" &&
          processed > 0 &&
          total >= processed
        ) {
          const elapsedSeconds = (Date.now() - transcribeStartedAtRef.current) / 1000;
          const averageChunkSeconds = elapsedSeconds / processed;
          const remainingChunks = Math.max(total - processed, 0);
          const estimatedRemaining = Math.ceil(averageChunkSeconds * remainingChunks);
          setEtaSeconds(estimatedRemaining);
        } else {
          setEtaSeconds(null);
        }
        return;
      }

      if (message.type === "partial") {
        if (message.requestId !== activeRequestIdRef.current) return;
        setStatus("transcribing");
        setOutput(message.text);
        return;
      }

      if (message.type === "segments") {
        if (message.requestId !== activeRequestIdRef.current) return;
        setSegments(message.segments);
        setOutput(message.text);
        setStatus("ready");
        setProgressPhase(null);
        setProcessedChunks(null);
        setTotalChunks(null);
        setEtaSeconds(null);
        transcribeStartedAtRef.current = null;
        return;
      }

      if (message.type === "result") {
        if (message.requestId !== activeRequestIdRef.current) return;
        setOutput(message.text);
        setStatus("ready");
        setProgressPhase(null);
        setProcessedChunks(null);
        setTotalChunks(null);
        setEtaSeconds(null);
        transcribeStartedAtRef.current = null;
        return;
      }

      if (message.type === "error") {
        if (
          typeof message.requestId === "number" &&
          message.requestId !== activeRequestIdRef.current
        ) {
          return;
        }
        setStatus("error");
        setError(message.error);
        clearProgressState();
      }
    },
    [clearProgressState],
  );

  const initializeWorker = useCallback(() => {
    if (typeof Worker === "undefined") {
      queueMicrotask(() => {
        setStatus("error");
        setError("Web Workers are not supported in this browser.");
      });
      return null;
    }

    const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      handleWorkerMessage(event.data);
    };

    worker.onerror = (event) => {
      setStatus("error");
      setError(event.message || "Worker encountered an unexpected error.");
      clearProgressState();
    };

    const loadRequest: WorkerRequest = { type: "load" };
    worker.postMessage(loadRequest);
    return worker;
  }, [clearProgressState, handleWorkerMessage]);

  const cancelTranscription = useCallback(() => {
    activeRequestIdRef.current += 1;
    setIsCancelling(true);
    setStatus("idle");
    setError(null);
    clearProgressState();

    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }

    initializeWorker();

    window.setTimeout(() => {
      setIsCancelling(false);
    }, 300);
  }, [clearProgressState, initializeWorker]);

  useEffect(() => {
    initializeWorker();

    return () => {
      if (copyResetTimeoutRef.current !== null) {
        window.clearTimeout(copyResetTimeoutRef.current);
      }
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [initializeWorker]);

  useEffect(() => {
    if (!isExportMenuOpen) return;

    const handleOutsideClick = (event: MouseEvent) => {
      if (
        exportMenuRef.current &&
        event.target instanceof Node &&
        !exportMenuRef.current.contains(event.target)
      ) {
        setIsExportMenuOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsExportMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isExportMenuOpen]);

  useEffect(() => {
    const textarea = outputTextareaRef.current;
    if (!textarea) return;

    textarea.style.height = "220px";
    const nextHeight = Math.max(220, Math.min(textarea.scrollHeight, 520));
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > 520 ? "auto" : "hidden";
  }, [fontMode, output]);

  useEffect(() => {
    if (progressPhase !== "transcribing" || etaSeconds === null || etaSeconds <= 0) {
      return;
    }

    const countdown = window.setInterval(() => {
      setEtaSeconds((current) => {
        if (current === null || current <= 0) return 0;
        return current - 1;
      });
    }, 1000);

    return () => {
      window.clearInterval(countdown);
    };
  }, [etaSeconds, progressPhase]);

  const handleFileSelected = useCallback(
    async (file: File) => {
      if (status === "loading" || status === "transcribing" || status === "decoding") {
        cancelTranscription();
      }

      const worker = workerRef.current;
      if (!worker) {
        setStatus("error");
        setError("Transcription worker is not available.");
        return;
      }

      const requestId = activeRequestIdRef.current + 1;
      activeRequestIdRef.current = requestId;

      setActiveFileName(file.name);
      setOutput("");
      setSegments([]);
      setCopyState("idle");
      setCopyFeedback(null);
      setIsExportMenuOpen(false);
      setError(null);
      setStatus("decoding");
      clearProgressState();

      try {
        const audioData = await decodeAudioFile(file);
        if (requestId !== activeRequestIdRef.current) return;

        const request: WorkerRequest = {
          type: "transcribe",
          requestId,
          audio: audioData,
          language: selectedLanguage,
        };
        worker.postMessage(request, [audioData.buffer]);
      } catch (decodeError) {
        if (requestId !== activeRequestIdRef.current) return;
        setStatus("error");
        setError(
          decodeError instanceof Error
            ? decodeError.message
            : "Failed to decode the selected audio file.",
        );
      }
    },
    [cancelTranscription, clearProgressState, selectedLanguage, status],
  );

  const plainTextExport = useMemo(() => {
    if (segments.length > 0) {
      return segments
        .map((segment) => segment.text.trim())
        .filter(Boolean)
        .join("\n\n");
    }
    return output.trim();
  }, [output, segments]);

  const timestampedExport = useMemo(() => {
    if (segments.length > 0) {
      return segments
        .map((segment) => `[${formatSegmentTimestamp(segment.start)}] ${segment.text}`)
        .join("\n");
    }
    return output.trim();
  }, [output, segments]);

  const hasExportContent = useMemo(
    () => plainTextExport.trim().length > 0 || timestampedExport.trim().length > 0,
    [plainTextExport, timestampedExport],
  );

  const buildJsonExportPayload = useCallback((): TranscriptExportJson => {
    return {
      version: 1,
      createdAt: new Date().toISOString(),
      fileName: activeFileName ?? null,
      model: "Xenova/whisper-small",
      language: selectedLanguage,
      text: output,
      segments,
    };
  }, [activeFileName, output, segments, selectedLanguage]);

  const triggerDownload = useCallback((content: string, filename: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }, []);

  const writeToClipboard = useCallback(
    async (value: string, successMessage: string) => {
      if (!value.trim()) return;

      try {
        if (!navigator.clipboard) {
          throw new Error("Clipboard API is unavailable in this browser.");
        }

        await navigator.clipboard.writeText(value);
        setCopyState("success");
        setCopyFeedback(successMessage);
      } catch (copyError) {
        setCopyState("error");
        setCopyFeedback("Copy failed");
        setError(
          copyError instanceof Error
            ? copyError.message
            : "Failed to copy transcript to clipboard.",
        );
      }

      if (copyResetTimeoutRef.current !== null) {
        window.clearTimeout(copyResetTimeoutRef.current);
      }

      copyResetTimeoutRef.current = window.setTimeout(() => {
        setCopyState("idle");
        setCopyFeedback(null);
      }, 1500);
    },
    [],
  );

  const handleSmartExport = useCallback(
    async (action: SmartExportAction) => {
      setIsExportMenuOpen(false);
      if (!hasExportContent) return;

      if (action === "copy_text_only") {
        await writeToClipboard(plainTextExport, "Text copied");
        return;
      }

      if (action === "copy_with_timestamps") {
        await writeToClipboard(timestampedExport, "Timestamped copy ready");
        return;
      }

      const jsonPayload = buildJsonExportPayload();
      triggerDownload(
        `${JSON.stringify(jsonPayload, null, 2)}\n`,
        `transcription-${timestampForFilename(new Date())}.json`,
        "application/json;charset=utf-8",
      );
    },
    [
      buildJsonExportPayload,
      hasExportContent,
      plainTextExport,
      timestampedExport,
      triggerDownload,
      writeToClipboard,
    ],
  );

  const handleDownloadTxt = useCallback(() => {
    if (!timestampedExport.trim()) return;

    triggerDownload(
      timestampedExport,
      `transcription-${timestampForFilename(new Date())}.txt`,
      "text/plain;charset=utf-8",
    );
  }, [timestampedExport, triggerDownload]);

  const statusChipClass = useMemo(() => {
    if (status === "error") return "border-red-500/50 bg-red-500/10 text-red-200";
    if (status === "ready") return "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
    if (status === "transcribing") return "border-cyan-400/40 bg-cyan-400/10 text-cyan-100";
    if (status === "decoding") return "border-blue-400/40 bg-blue-400/10 text-blue-100";
    if (status === "loading") return "border-amber-400/40 bg-amber-400/10 text-amber-100";
    return "border-neutral-600/70 bg-neutral-800/70 text-neutral-200";
  }, [status]);

  const progressLabel = useMemo(() => {
    if (progressPhase === "download") {
      if (downloadedBytes !== null && totalBytes !== null && totalBytes > 0) {
        const dlMB = (downloadedBytes / (1024 * 1024)).toFixed(1);
        const totalMB = (totalBytes / (1024 * 1024)).toFixed(1);
        return `Downloading model... ${dlMB} MB / ${totalMB} MB (${progress.toFixed(0)}%)`;
      }
      return `Downloading model... ${progress.toFixed(0)}%`;
    }
    if (progressPhase === "transcribing") {
      if (processedChunks !== null && totalChunks !== null) {
        return `Transcribing... ${progress.toFixed(0)}% (${processedChunks}/${totalChunks} chunks)`;
      }
      return `Transcribing... ${progress.toFixed(0)}%`;
    }
    return "";
  }, [downloadedBytes, processedChunks, progress, progressPhase, totalBytes, totalChunks]);

  const etaLabel = useMemo(() => {
    if (progressPhase !== "transcribing") {
      return null;
    }
    if (etaSeconds === null) {
      return "Estimated time left: calculating...";
    }
    if (etaSeconds <= 0) {
      return "Estimated time left: finishing up...";
    }
    return `Estimated time left: ${formatSegmentTimestamp(etaSeconds)}`;
  }, [etaSeconds, progressPhase]);

  const busy =
    status === "loading" || status === "decoding" || status === "transcribing" || isCancelling;
  const showProgressBar = progressPhase === "download" || progressPhase === "transcribing";
  const showSkeleton =
    !output && (status === "loading" || status === "decoding" || status === "transcribing");

  const placeholderText =
    status === "loading"
      ? "Downloading and initializing Whisper Small. This may take a moment on first run — the model is cached afterwards."
      : status === "decoding"
        ? "Decoding and resampling audio to 16kHz..."
        : status === "transcribing"
          ? "Analyzing audio and generating transcript..."
          : status === "error"
            ? "Transcription failed. Please try another file."
            : "Upload audio to start a local transcription.";

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-10 sm:px-6">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(34,211,238,0.14),transparent_40%),radial-gradient(circle_at_80%_90%,rgba(59,130,246,0.12),transparent_35%)]" />

      <section className="relative w-full max-w-4xl rounded-2xl border border-white/10 bg-neutral-900/70 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_24px_80px_rgba(0,0,0,0.55)] backdrop-blur-sm sm:p-8">
        <header className="mb-8 space-y-3">
          <p className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">
            <ShieldCheck className="size-3.5" />
            100% local and private
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            Client-Side Audio Transcription
          </h1>
          <p className="max-w-2xl text-sm leading-6 text-neutral-300 sm:text-base">
            Upload lecture or meeting audio to generate transcripts directly in your browser.
            No server uploads, no third-party processing.
          </p>
        </header>

        <UploadDropzone onFileSelected={handleFileSelected} />

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <div
            className={[
              "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium",
              statusChipClass,
            ].join(" ")}
          >
            {status === "loading" || status === "decoding" || status === "transcribing" ? (
              <LoaderCircle className="size-3.5 animate-spin" />
            ) : null}
            <span>{statusLabel(status)}</span>
          </div>
          <p className="text-xs text-neutral-400 sm:text-sm">
            {activeFileName ? `Current file: ${activeFileName}` : "No file selected"}
          </p>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <label
            htmlFor="language-select"
            className="text-xs font-medium uppercase tracking-wide text-neutral-400"
          >
            Language
          </label>
          <select
            id="language-select"
            value={selectedLanguage}
            onChange={(event) => {
              setSelectedLanguage(event.target.value as "auto" | WhisperLanguage);
            }}
            className="rounded-lg border border-white/10 bg-neutral-950/80 px-3 py-1.5 text-sm text-neutral-200 outline-none transition-colors focus:border-cyan-400/60"
          >
            {LANGUAGE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {showProgressBar ? (
          <div className="mt-3 space-y-2 rounded-xl border border-white/10 bg-neutral-950/70 p-3">
            {progressPhase === "download" ? (
              <p className="text-xs text-amber-300/80">
                ⬇ First run only — model will be cached in your browser after this download.
              </p>
            ) : null}
            <div className="flex items-center justify-between gap-4">
              <div className="space-y-0.5">
                <p className="text-xs text-neutral-300 sm:text-sm">{progressLabel}</p>
                {etaLabel ? <p className="text-xs text-neutral-500">{etaLabel}</p> : null}
              </div>
              <button
                type="button"
                onClick={cancelTranscription}
                disabled={isCancelling}
                className="inline-flex items-center gap-1.5 rounded-md border border-red-500/40 bg-red-500/10 px-2.5 py-1 text-xs font-medium text-red-200 transition-colors hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Square className="size-3.5" />
                {isCancelling ? "Cancelling..." : "Cancel"}
              </button>
            </div>
            <div className="h-2 overflow-hidden rounded-full border border-white/10 bg-neutral-900/90">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-300 to-blue-400 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        ) : null}

        {status === "decoding" && !showProgressBar ? (
          <div className="mt-3 flex items-center justify-between rounded-xl border border-white/10 bg-neutral-950/70 p-3">
            <p className="text-xs text-neutral-300 sm:text-sm">Decoding audio...</p>
            <button
              type="button"
              onClick={cancelTranscription}
              disabled={isCancelling}
              className="inline-flex items-center gap-1.5 rounded-md border border-red-500/40 bg-red-500/10 px-2.5 py-1 text-xs font-medium text-red-200 transition-colors hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <Square className="size-3.5" />
              {isCancelling ? "Cancelling..." : "Cancel"}
            </button>
          </div>
        ) : null}

        {error ? (
          <div className="mt-3 inline-flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle className="size-4" />
            <span>{error}</span>
          </div>
        ) : null}

        <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-neutral-950/75">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-4 py-2">
            <div className="flex items-center gap-3">
              <p className="text-sm font-medium text-neutral-200">Transcript Output</p>
              {output ? (
                <span className="text-xs text-neutral-500">
                  {output.trim().split(/\s+/).filter(Boolean).length} words
                </span>
              ) : null}
            </div>
            <div className="flex items-center gap-2">
              <div ref={exportMenuRef} className="relative">
                <button
                  type="button"
                  onClick={() => setIsExportMenuOpen((prev) => !prev)}
                  disabled={!hasExportContent}
                  aria-haspopup="menu"
                  aria-expanded={isExportMenuOpen}
                  className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-200 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Copy className="size-3.5" />
                  Smart Export
                  <ChevronDown
                    className={[
                      "size-3.5 transition-transform",
                      isExportMenuOpen ? "rotate-180" : "",
                    ].join(" ")}
                  />
                </button>

                {isExportMenuOpen ? (
                  <div
                    role="menu"
                    aria-label="Smart export actions"
                    className="absolute right-0 z-20 mt-2 w-56 rounded-lg border border-white/10 bg-neutral-900 p-1 shadow-xl"
                  >
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => void handleSmartExport("copy_text_only")}
                      className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs text-neutral-200 transition-colors hover:bg-neutral-800"
                    >
                      <Copy className="size-3.5" />
                      Copy Text Only
                    </button>
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => void handleSmartExport("copy_with_timestamps")}
                      className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs text-neutral-200 transition-colors hover:bg-neutral-800"
                    >
                      <Clock3 className="size-3.5" />
                      Copy with Timestamps
                    </button>
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => void handleSmartExport("export_json")}
                      className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs text-neutral-200 transition-colors hover:bg-neutral-800"
                    >
                      <FileJson className="size-3.5" />
                      Export to JSON
                    </button>
                  </div>
                ) : null}
              </div>

              <button
                type="button"
                onClick={handleDownloadTxt}
                disabled={!timestampedExport.trim()}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-200 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Download className="size-3.5" />
                Download .txt
              </button>

              <button
                type="button"
                onClick={() => setFontMode((prev) => (prev === "sans" ? "mono" : "sans"))}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-200 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800"
              >
                <Type className="size-3.5" />
                {fontMode === "sans" ? "Mono" : "Sans"}
              </button>
            </div>
          </div>

          <div className="px-4 pt-2">
            {copyFeedback ? (
              <p
                className={[
                  "mb-2 inline-flex items-center gap-1.5 text-xs",
                  copyState === "error" ? "text-red-300" : "text-emerald-300",
                ].join(" ")}
              >
                {copyState === "success" ? <Check className="size-3.5" /> : null}
                {copyFeedback}
              </p>
            ) : null}
          </div>

          <div className="p-4 pt-0">
            {showSkeleton ? (
              <div className="min-h-[220px] animate-pulse space-y-3 rounded-lg border border-white/5 bg-neutral-900/40 p-4">
                <div className="h-3 w-11/12 rounded bg-neutral-800" />
                <div className="h-3 w-10/12 rounded bg-neutral-800" />
                <div className="h-3 w-9/12 rounded bg-neutral-800" />
                <div className="h-3 w-8/12 rounded bg-neutral-800" />
                <div className="h-3 w-11/12 rounded bg-neutral-800" />
                <div className="h-3 w-7/12 rounded bg-neutral-800" />
              </div>
            ) : segments.length > 0 ? (
              <div className="max-h-[520px] overflow-y-auto rounded-lg border border-white/10 bg-neutral-900/50">
                <ul className="divide-y divide-white/5">
                  {segments.map((segment, index) => (
                    <li
                      key={`${segment.start}-${segment.end}-${index}`}
                      className="grid grid-cols-[80px_1fr] gap-3 px-3 py-2.5"
                    >
                      <span className="pt-0.5 text-xs text-neutral-500">
                        [{formatSegmentTimestamp(segment.start)}]
                      </span>
                      <p
                        className={[
                          "text-sm leading-6 text-neutral-200",
                          fontMode === "mono" ? "font-mono" : "font-sans",
                        ].join(" ")}
                      >
                        {segment.text}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <textarea
                ref={outputTextareaRef}
                readOnly
                value={output}
                placeholder={placeholderText}
                className={[
                  "w-full min-h-[220px] max-h-[520px] resize-none rounded-lg border border-white/10 bg-neutral-900/60 p-4 text-sm leading-6 text-neutral-200 outline-none",
                  fontMode === "mono" ? "font-mono" : "font-sans",
                ].join(" ")}
              />
            )}
          </div>
        </div>

        <div className="mt-4 rounded-lg border border-white/10 bg-neutral-950/70 px-3 py-2 text-xs text-neutral-400 sm:text-sm">
          Supports <span className="font-medium text-neutral-300">.mp3, .wav, .m4a, .mp4, .ogg</span>.
          Transcription runs in-browser with{" "}
          <span className="font-medium text-neutral-300">Xenova/whisper-small</span>.
        </div>

        {busy ? (
          <p className="mt-2 text-xs text-neutral-500">
            Leave this tab open while processing long lectures for the best result stability.
          </p>
        ) : null}
      </section>
    </main>
  );
}
