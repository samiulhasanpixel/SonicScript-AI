"use client";

import {
  AlertCircle,
  Check,
  ChevronDown,
  Clock3,
  Copy,
  Cpu,
  FileJson,
  Linkedin,
  Lock,
  Monitor,
  ShieldCheck,
  Square,
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

type LanguageOption = { value: "auto" | WhisperLanguage; label: string; flag: string };
type ProgressPhase = "download" | "transcribing";
type CopyState = "idle" | "success" | "error";
type SmartExportAction = "copy_text_only" | "copy_with_timestamps" | "export_json";

type WorkerStatus = "loading" | "ready" | "transcribing" | "error";
type TranscriptionStatus = "idle" | "loading" | "decoding" | "transcribing" | "ready" | "error";

type WorkerRequest =
  | { type: "load"; model?: string }
  | { type: "cancel"; requestId: number }
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
  language: "auto" | WhisperLanguage | null;
  text: string;
  segments: TranscriptSegment[];
};

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

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { value: "english", label: "English", flag: "🇬🇧" },
  { value: "turkish", label: "Turkish", flag: "🇹🇷" },
  { value: "spanish", label: "Spanish", flag: "🇪🇸" },
  { value: "french", label: "French", flag: "🇫🇷" },
  { value: "german", label: "German", flag: "🇩🇪" },
  { value: "italian", label: "Italian", flag: "🇮🇹" },
  { value: "portuguese", label: "Portuguese", flag: "🇵🇹" },
  { value: "russian", label: "Russian", flag: "🇷🇺" },
  { value: "arabic", label: "Arabic", flag: "🇸🇦" },
  { value: "hindi", label: "Hindi", flag: "🇮🇳" },
  { value: "japanese", label: "Japanese", flag: "🇯🇵" },
  { value: "korean", label: "Korean", flag: "🇰🇷" },
];

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

  const audioContext = new AudioContextClass({ sampleRate: 16_000 });
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const mono = downmixToMono(audioBuffer);
    return resampleMonoAudio(mono, audioBuffer.sampleRate, 16_000);
  } finally {
    await audioContext.close();
  }
}

function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i += 1) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // 1 channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (sampleRate * block align)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
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
  const langMenuRef = useRef<HTMLDivElement | null>(null);
  const transcribeStartedAtRef = useRef<number | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

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
  const [selectedLanguage, setSelectedLanguage] = useState<"auto" | WhisperLanguage | null>(null);
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const [isLangMenuOpen, setIsLangMenuOpen] = useState(false);
  const [isLangShaking, setIsLangShaking] = useState(false);
  const [isModelShaking, setIsModelShaking] = useState(false);
  const [justCompleted, setJustCompleted] = useState(false);
  const [downloadedBytes, setDownloadedBytes] = useState<number | null>(null);
  const [totalBytes, setTotalBytes] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"plain" | "timestamps">("plain");
  const [loadingDetail, setLoadingDetail] = useState<string | null>(null);
  const [activeDevice, setActiveDevice] = useState<"webgpu" | "wasm" | null>(null);
  const [currentSlice, setCurrentSlice] = useState<number | null>(null);
  const [totalSlices, setTotalSlices] = useState<number | null>(null);
  const [warmUpElapsed, setWarmUpElapsed] = useState(0);
  const [isMobile, setIsMobile] = useState(false);
  const [gpuSupported, setGpuSupported] = useState(true);
  const [isViaCloud, setIsViaCloud] = useState(false);
  // True if the model was successfully loaded in a previous session.
  // Stored in localStorage so page refresh doesn't re-show the loading UI.
  const [wasModelEverLoaded, setWasModelEverLoaded] = useState(false);

  useEffect(() => {
    setIsMobile(/iPhone|iPad|iPod|Android/i.test(navigator.userAgent));
    setGpuSupported("gpu" in navigator);
    // Check if model was loaded in a previous browser session.
    if (typeof window !== "undefined" && localStorage.getItem("whisper_model_cached") === "1") {
      setWasModelEverLoaded(true);
    }
  }, []);

  // Kick off model preload as soon as the user picks a language.
  // The worker's getTranscriber() caches the pipeline, so when the
  // user later drops a file the model is already ready (or still
  // downloading — in which case the transcribe request just awaits).
  // If the model is already "ready" (previously loaded), skip re-sending
  // the load message so we don't cause a brief loading flash.
  useEffect(() => {
    if (!selectedLanguage || isMobile) return;
    const worker = workerRef.current;
    if (!worker) return;
    // Only send a load request when truly idle (never loaded yet).
    // If the model is already ready/loaded, changing language doesn't
    // require reloading — Whisper language is passed per-transcription call.
    if (status !== "idle") return;
    const loadRequest: WorkerRequest = { type: "load" };
    worker.postMessage(loadRequest);
  }, [selectedLanguage, isMobile, status]);



  // Flash "Transcription complete" badge only when transitioning transcribing → ready
  const prevStatusRef = useRef<typeof status | null>(null);
  useEffect(() => {
    if (prevStatusRef.current === "transcribing" && status === "ready") {
      setJustCompleted(true);
      const id = window.setTimeout(() => setJustCompleted(false), 2800);
      prevStatusRef.current = status;
      return () => window.clearTimeout(id);
    }
    prevStatusRef.current = status;
  }, [status]);

  const clearProgressState = useCallback(() => {
    setProgress(0);
    setProgressPhase(null);
    setProcessedChunks(null);
    setTotalChunks(null);
    setEtaSeconds(null);
    setDownloadedBytes(null);
    setTotalBytes(null);
    setLoadingDetail(null);
    setCurrentSlice(null);
    setTotalSlices(null);
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
          if (message.detail) setLoadingDetail(message.detail);
          if (message.device) setActiveDevice(message.device as "webgpu" | "wasm");
        } else if (message.status === "transcribing") {
          setStatus("transcribing");
          setLoadingDetail(null);
          setProgress(0);
          setProgressPhase("transcribing");
          if (message.device) setActiveDevice(message.device as "webgpu" | "wasm");
        } else if (message.status === "ready") {
          setStatus("ready");
          setLoadingDetail(null);
          if (message.device) setActiveDevice(message.device as "webgpu" | "wasm");
          setProgressPhase(null);
          setProcessedChunks(null);
          setTotalChunks(null);
          setEtaSeconds(null);
          transcribeStartedAtRef.current = null;
          // Persist that the model has been loaded at least once so future
          // page loads can suppress the brief loading UI on reinitialisation.
          setWasModelEverLoaded(true);
          if (typeof window !== "undefined") {
            localStorage.setItem("whisper_model_cached", "1");
          }
        } else if (message.status === "error") {
          setStatus("error");
          setLoadingDetail(null);
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
        if (typeof message.currentSlice === "number") setCurrentSlice(message.currentSlice);
        if (typeof message.totalSlices === "number") setTotalSlices(message.totalSlices);

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

    return worker;
  }, [clearProgressState, handleWorkerMessage]);

  const cancelTranscription = useCallback(() => {
    const currentRequestId = activeRequestIdRef.current;
    activeRequestIdRef.current += 1;
    setIsCancelling(true);
    setError(null);
    clearProgressState();

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsViaCloud(false);
      setStatus("ready");
      setActiveFileName(null);
    }

    const isModelLoaded =
      status === "ready" || status === "transcribing" || status === "decoding";

    if (isModelLoaded && !isViaCloud) {
      // Model is already in memory — just abort the running transcription.
      // No need to kill the worker; the loaded pipeline stays intact.
      if (workerRef.current) {
        const cancelRequest: WorkerRequest = { type: "cancel", requestId: currentRequestId };
        (workerRef.current as Worker).postMessage(cancelRequest);
      }
      setStatus("ready");
      setActiveFileName(null);
    } else {
      // Model wasn't loaded yet — terminate and let it restart cleanly.
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
      const newWorker = initializeWorker();
      if (selectedLanguage && newWorker) {
        const loadRequest: WorkerRequest = { type: "load" };
        newWorker.postMessage(loadRequest);
      }
      setStatus("idle");
    }

    window.setTimeout(() => {
      setIsCancelling(false);
    }, 300);
  }, [clearProgressState, initializeWorker, selectedLanguage, status]);

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
    if (!isLangMenuOpen) return;

    const handleOutsideClick = (event: MouseEvent) => {
      if (
        langMenuRef.current &&
        event.target instanceof Node &&
        !langMenuRef.current.contains(event.target)
      ) {
        setIsLangMenuOpen(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsLangMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isLangMenuOpen]);

  useEffect(() => {
    const textarea = outputTextareaRef.current;
    if (!textarea) return;

    textarea.style.height = "220px";
    const nextHeight = Math.max(220, Math.min(textarea.scrollHeight, 520));
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > 520 ? "auto" : "hidden";

    // Auto-scroll to the bottom while transcription is streaming so the user
    // can watch new chunks appear live without manually scrolling.
    if (status === "transcribing") {
      textarea.scrollTop = textarea.scrollHeight;
    }
  }, [output, status]);

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

  const startTranscription = useCallback(
    async (file: File) => {
      // Don't cancel/restart the worker if we're only preloading the model
      // (status === "loading" with no active file). Only cancel if a previous
      // transcription is already in flight.
      if (status === "transcribing" || status === "decoding") {
        cancelTranscription();
      }

      if (!isMobile) {
        const worker = workerRef.current;
        if (!worker) {
          setStatus("error");
          setError("Transcription worker is not available.");
          return;
        }
      }

      const requestId = activeRequestIdRef.current + 1;
      activeRequestIdRef.current = requestId;

      setActiveFileName(file.name);
      setOutput("");
      setSegments([]);
      setViewMode("plain");
      setCopyState("idle");
      setCopyFeedback(null);
      setIsExportMenuOpen(false);
      setError(null);
      clearProgressState();

      if (isMobile) {
        setIsViaCloud(true);
        setStatus("transcribing");
        setProgressPhase("transcribing");
        setLoadingDetail("Uploading to Cloud...");
        setProgress(0);

        const controller = new AbortController();
        abortControllerRef.current = controller;

        try {
          const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
          let chunksToProcess: { blob: Blob; offsetS: number }[] = [];

          if (file.size <= MAX_FILE_SIZE) {
            chunksToProcess = [{ blob: file, offsetS: 0 }];
            setTotalChunks(1);
          } else {
            setLoadingDetail("File too large, chunking audio locally...");
            setStatus("decoding"); // visually update
            const audioData = await decodeAudioFile(file);
            setStatus("transcribing"); // back to transcribing

            const CHUNK_DURATION_S = 10 * 60; // 10 minutes
            const SAMPLES_PER_CHUNK = CHUNK_DURATION_S * 16000;

            for (let i = 0; i < audioData.length; i += SAMPLES_PER_CHUNK) {
              const chunkData = audioData.slice(i, i + SAMPLES_PER_CHUNK);
              const wavBlob = encodeWAV(chunkData, 16000);
              chunksToProcess.push({ blob: wavBlob, offsetS: i / 16000 });
            }
            setTotalChunks(chunksToProcess.length);
          }

          let combinedText = "";
          const combinedSegments: TranscriptSegment[] = [];

          for (let i = 0; i < chunksToProcess.length; i++) {
            if (abortControllerRef.current?.signal.aborted) break;

            const { blob, offsetS } = chunksToProcess[i];
            setLoadingDetail(`Uploading chunk ${i + 1} of ${chunksToProcess.length}...`);
            setProcessedChunks(i);
            setProgress((i / chunksToProcess.length) * 100);

            const formData = new FormData();
            formData.append("file", blob, `chunk-${i}.wav`);
            if (selectedLanguage && selectedLanguage !== "auto") {
              formData.append("language", selectedLanguage);
            }

            const response = await fetch("/api/transcribe", {
              method: "POST",
              body: formData,
              signal: controller.signal,
            });

            if (!response.ok) {
              const data = await response.json().catch(() => ({}));
              throw new Error(data.error || "Cloud transcription failed.");
            }

            if (requestId !== activeRequestIdRef.current) return;

            const result = await response.json();

            combinedText += (combinedText ? " " : "") + (result.text || "").trim();

            if (result.segments) {
              for (const seg of result.segments) {
                combinedSegments.push({
                  text: seg.text,
                  start: seg.start + offsetS,
                  end: seg.end + offsetS,
                });
              }
            }
          }

          if (abortControllerRef.current?.signal.aborted) return;
          if (requestId !== activeRequestIdRef.current) return;

          setProgress(100);
          setProcessedChunks(chunksToProcess.length);
          setOutput(combinedText);
          setSegments(combinedSegments);
          setStatus("ready");
          setProgressPhase(null);
          setIsViaCloud(false);
          abortControllerRef.current = null;

        } catch (cloudError: unknown) {
          if (cloudError instanceof Error && cloudError.name === "AbortError") return;
          if (requestId !== activeRequestIdRef.current) return;
          setStatus("error");
          setError(cloudError instanceof Error ? cloudError.message : "Cloud transcription failed.");
          setIsViaCloud(false);
          abortControllerRef.current = null;
        }
        return;
      }

      setStatus("decoding");

      try {
        const audioData = await decodeAudioFile(file);
        if (requestId !== activeRequestIdRef.current) return;

        const request: WorkerRequest = {
          type: "transcribe",
          requestId,
          audio: audioData,
          language: selectedLanguage ?? "english",
        };
        workerRef.current?.postMessage(request, [audioData.buffer]);
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
    [cancelTranscription, clearProgressState, isMobile, selectedLanguage, status],
  );

  const handleFileSelected = useCallback(
    (file: File) => {
      if (!selectedLanguage) return;
      startTranscription(file);
    },
    [selectedLanguage, startTranscription],
  );

  const plainTextExport = useMemo(() => {
    if (segments.length > 0) {
      return segments
        .map((segment) => segment.text.trim())
        .filter(Boolean)
        .join(" ");
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

  const progressLabel = useMemo(() => {
    if (progressPhase === "download") {
      if (downloadedBytes !== null && totalBytes !== null && totalBytes > 0) {
        const totalMB = (totalBytes / (1024 * 1024)).toFixed(1);
        const dlMB = (downloadedBytes / (1024 * 1024)).toFixed(1).padStart(totalMB.length, "\u00A0");
        const pct = progress.toFixed(0).padStart(3, "\u00A0");
        return `Downloading model\u2026 ${dlMB} / ${totalMB} MB (${pct}%)`;
      }
      const pct = progress.toFixed(0).padStart(3, "\u00A0");
      return `Downloading model\u2026 ${pct}%`;
    }
    if (progressPhase === "transcribing") {
      // Processed audio time: each 30-s chunk with 10-s jump = 20 s of new audio per chunk
      const processedAudioSec =
        processedChunks !== null ? Math.round(processedChunks * 20) : null;
      const totalAudioSec =
        totalChunks !== null ? Math.round((totalChunks - 1) * 20 + 30) : null;

      const fmtMin = (s: number) => {
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return sec === 0 ? `${m} min` : `${m}:${String(sec).padStart(2, "0")} min`;
      };

      const timeStr =
        processedAudioSec !== null && totalAudioSec !== null
          ? `${fmtMin(processedAudioSec)} / ${fmtMin(totalAudioSec)} transcribed`
          : null;

      const sliceStr =
        totalSlices !== null && totalSlices > 1 && currentSlice !== null
          ? `Slice ${currentSlice}/${totalSlices}`
          : null;

      const pctStr = `${progress.toFixed(0)}%`;

      return [sliceStr, timeStr, pctStr].filter(Boolean).join("  ·  ");
    }
    return "";
  }, [currentSlice, downloadedBytes, processedChunks, progress, progressPhase, totalBytes, totalChunks, totalSlices]);

  /**
   * Rough audio duration in minutes derived from total chunk count.
   * Each chunk advances by (CHUNK_LENGTH_S - 2 * STRIDE_LENGTH_S) = 20 s,
   * so total audio ≈ (chunks − 1) × 20s + 30s.
   */
  const roughAudioMinutes =
    totalChunks !== null && totalChunks > 0
      ? Math.round(((totalChunks - 1) * 20 + 30) / 60)
      : null;

  const etaLabel = useMemo(() => {
    if (progressPhase !== "transcribing") return null;

    // Before the first chunk completes we have no timing data —
    // show a rough estimate from audio duration instead of "calculating..."
    if (etaSeconds === null) {
      if (roughAudioMinutes !== null && roughAudioMinutes > 0) {
        // whisper-small on WebGPU processes roughly 3–5× real-time
        const low = Math.max(1, Math.round(roughAudioMinutes / 5));
        const high = Math.max(2, Math.round(roughAudioMinutes / 2));
        return `Audio length ~${roughAudioMinutes} min — estimated processing time: ${low}–${high} min`;
      }
      return "Estimated time: calculating...";
    }
    if (etaSeconds <= 0) return "Estimated time: finishing...";
    return `Estimated remaining: ${formatSegmentTimestamp(etaSeconds)}`;
  }, [etaSeconds, progressPhase, roughAudioMinutes]);

  const busy =
    status === "loading" || status === "decoding" || status === "transcribing" || isCancelling;
  // Only show the compact file row (instead of dropzone) when there's an active file being processed.
  // During model preloading (status === "loading", activeFileName === null) we keep the dropzone visible.
  const uploadBusy = busy && activeFileName !== null;
  /** True once the model pipeline is loaded and ready to transcribe. */
  // Suppress the loading UI on page-refresh if the model was previously cached.
  // Only show the loading UI again if bytes are actually being transferred
  // (i.e. the browser cache was cleared and a real re-download is happening).
  const isActuallyDownloading =
    progressPhase === "download" &&
    typeof totalBytes === "number" &&
    totalBytes > 0 &&
    typeof downloadedBytes === "number" &&
    downloadedBytes > 0;
  const modelReady =
    status === "ready" ||
    status === "transcribing" ||
    status === "decoding" ||
    isMobile ||
    (wasModelEverLoaded && status === "loading" && !isActuallyDownloading);
  const isCompiling = status === "loading" && loadingDetail === "compiling";
  /** True between "transcribing" status and the very first chunk_callback firing. */
  const isWarmingUp =
    status === "transcribing" &&
    processedChunks === 0 &&
    totalChunks !== null &&
    totalChunks > 0;

  // Live elapsed-seconds counter while the GPU processes the very first chunk.
  // This is the ONLY visual proof of activity during an otherwise silent 30-90 s wait.
  useEffect(() => {
    if (!isWarmingUp) {
      setWarmUpElapsed(0);
      return;
    }
    setWarmUpElapsed(0);
    const id = window.setInterval(() => setWarmUpElapsed((prev) => prev + 1), 1_000);
    return () => window.clearInterval(id);
  }, [isWarmingUp]);
  const showProgressBar = progressPhase === "transcribing" && !isWarmingUp;
  const showSkeleton =
    !output && (status === "loading" || status === "decoding" || status === "transcribing");

  const placeholderText =
    status === "loading"
      ? isCompiling
        ? "Compiling WebGPU shaders for first-time setup — this takes 1–2 minutes and is fully cached afterwards."
        : "Downloading Whisper model to your browser cache. This only happens once."
      : status === "decoding"
        ? "Decoding and resampling audio to 16kHz..."
        : status === "transcribing"
          ? "Analyzing audio and generating transcript..."
          : status === "error"
            ? "Transcription failed. Please try another file."
            : "Upload audio to start a local transcription.";

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-10 sm:px-6">

      <section className="relative w-full max-w-4xl rounded-2xl border border-white/10 bg-neutral-900/70 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_24px_80px_rgba(0,0,0,0.55)] backdrop-blur-sm sm:p-8">
        <header className="mb-8 space-y-3">

          <div className="flex items-center gap-4">
            <svg width="52" height="52" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0">
              <rect x="4" y="12" width="2" height="8" rx="1" fill="white" className="fill-white" />
              <rect x="8" y="6" width="2" height="20" rx="1" fill="white" className="fill-white" />
              <rect x="12" y="10" width="2" height="12" rx="1" fill="white" className="fill-white" />
              <rect x="18" y="14" width="10" height="2" rx="1" fill="white" className="fill-white" />
              <rect x="18" y="18" width="7" height="2" rx="1" fill="white" className="fill-white" />
            </svg>
            <div>
              <div className="mb-1.5 flex items-center gap-2">
                <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2.5 py-0.5 text-xs font-semibold text-emerald-300">
                  <svg xmlns="http://www.w3.org/2000/svg" height="11px" viewBox="0 -960 960 960" width="11px" fill="currentColor"><path d="m382-354 339-339q12-12 28-12t28 12q12 12 12 28.5T777-636L410-268q-12 12-28 12t-28-12L182-440q-12-12-11.5-28.5T183-497q12-12 28.5-12t28.5 12l142 143Z" /></svg>
                  100% Free
                </span>
                <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-xs font-medium text-neutral-400">
                  No sign-up required
                </span>
              </div>
              <h1 className="text-3xl font-bold tracking-tight text-white sm:text-5xl">
                Audio Transcription Tool
              </h1>
            </div>
          </div>
          <p className="max-w-2xl text-sm leading-6 text-neutral-300 sm:text-base">
            Free audio-to-text transcription powered by Whisper AI — runs directly in your browser.
            {isMobile ? " Secure cloud processing for mobile devices." : " No server uploads, no accounts, no limits."}
          </p>
        </header>

        {/* Info panel */}
        <div className="mb-8 flex flex-col divide-y divide-white/5 rounded-2xl border border-white/10 bg-neutral-900/40 sm:flex-row sm:divide-x sm:divide-y-0">
          <div className="flex flex-1 flex-col p-5">
            <div className="mb-3 flex items-center gap-2 text-neutral-300">
              <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M320-240q-33 0-56.5-23.5T240-320v-320q0-33 23.5-56.5T320-720h320q33 0 56.5 23.5T720-640v320q0 33-23.5 56.5T640-240H320Zm0-80h320v-320H320v320Zm-80 40v-80h-80v-80h80v-80h-80v-80h80v-80h-80v-80h80v-80h80v80h80v-80h80v80h80v-80h80v80h80v80h-80v80h80v80h-80v80h80v80h-80v80h-80v-80h-80v80h-80v-80h-80v80h-80Zm160-240h160v-160H400v160Zm0-80h160v-160H400v160Z" /></svg>
              <h3 className="text-sm font-medium text-neutral-200">{isMobile ? "Hybrid Intelligence" : "Runs in your browser"}</h3>
            </div>
            <p className="text-xs leading-relaxed text-neutral-400">
              {isMobile
                ? "Switching seamlessly between on-device decoding and cloud transcription for the best mobile experience."
                : "Powered by Whisper Small via WebGPU. No internet connection required after the initial model load."}
            </p>
          </div>

          <div className="flex flex-1 flex-col p-5">
            <div className="mb-3 flex items-center gap-2 text-neutral-300">
              <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M240-80q-33 0-56.5-23.5T160-160v-400q0-33 23.5-56.5T240-640h40v-80q0-83 58.5-141.5T480-920q83 0 141.5 58.5T680-720v80h40q33 0 56.5 23.5T800-560v400q0 33-23.5 56.5T720-80H240Zm0-80h480v-400H240v400Zm240-120q33 0 56.5-23.5T560-360q0-33-23.5-56.5T480-440q-33 0-56.5 23.5T400-360q0 33 23.5 56.5T480-280ZM360-640h240v-80q0-50-35-85t-85-35q-50 0-85 35t-35 85v80ZM240-160v-400 400Z" /></svg>
              <h3 className="text-sm font-medium text-neutral-200">{isMobile ? "Privacy Conscious" : "Zero data leaves device"}</h3>
            </div>
            <p className="text-xs leading-relaxed text-neutral-400">
              {isMobile
                ? "Audio is securely sent to Groq Cloud for processing and deleted immediately. No personal data is stored."
                : "Your audio is never uploaded to any server. Everything is processed locally with no tracking or storage."}
            </p>
          </div>

          <div className="flex flex-1 flex-col p-5">
            <div className={[
              "mb-3 flex items-center gap-2",
              isMobile ? "text-cyan-400" : "text-neutral-300"
            ].join(" ")}>
              <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M320-120v-80H160q-33 0-56.5-23.5T80-280v-480q0-33 23.5-56.5T160-840h640q33 0 56.5 23.5T880-760v480q0 33-23.5 56.5T800-200H640v80H320ZM160-280h640v-480H160v480Zm0 0v-480 480Z" /></svg>
              <h3 className={[
                "text-sm font-medium",
                isMobile ? "text-cyan-200" : "text-neutral-200"
              ].join(" ")}>
                {isMobile ? "Works great on mobile" : "Works on desktop & mobile"}
              </h3>
            </div>
            <p className="text-xs leading-relaxed text-neutral-400">
              {isMobile
                ? "Full transcription support via Groq Cloud — fast, accurate, and battery friendly."
                : "Use Chrome or Edge on desktop for local WebGPU processing. On mobile, Groq Cloud handles everything automatically."}
            </p>
          </div>
        </div>

        <div className="mb-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-neutral-400">
            Step 1 — Select the audio language
          </p>
          <div
            ref={langMenuRef}
            className={["relative inline-block", isLangShaking ? "lang-shake" : ""].join(" ")}
            onAnimationEnd={() => setIsLangShaking(false)}
          >
            <button
              type="button"
              onClick={() => setIsLangMenuOpen((prev) => !prev)}
              className={[
                "inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium outline-none transition-colors",
                selectedLanguage
                  ? "border-cyan-400/40 bg-cyan-400/5 text-neutral-200 hover:bg-cyan-400/10"
                  : "border-dashed border-white/20 bg-neutral-900/60 text-neutral-400 hover:border-white/40 hover:text-neutral-200",
              ].join(" ")}
            >
              {selectedLanguage ? (
                <>
                  <span className="text-base leading-none">
                    {LANGUAGE_OPTIONS.find((o) => o.value === selectedLanguage)?.flag ?? ""}
                  </span>
                  {LANGUAGE_OPTIONS.find((o) => o.value === selectedLanguage)?.label}
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="currentColor"><path d="m476-80 182-480h84L924-80h-84l-43-122H603L560-80h-84ZM160-200l-56-56 202-202q-35-35-63.5-80T190-640h84q20 39 40 68t48 58q33-33 68.5-92.5T484-720H40v-80h280v-80h80v80h280v80H564q-21 72-63 148t-83 116l96 98-30 82-97-99-202 195Zm468-72h144l-72-204-72 204Z" /></svg>
                  Select audio language
                </>
              )}
              <ChevronDown
                className={[
                  "size-3.5 transition-transform",
                  isLangMenuOpen ? "rotate-180" : "",
                ].join(" ")}
              />
            </button>

            {isLangMenuOpen ? (
              <div
                role="listbox"
                className="absolute left-0 z-20 mt-2 w-48 rounded-lg border border-white/10 bg-neutral-900 p-1 shadow-xl"
              >
                {LANGUAGE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    role="option"
                    aria-selected={selectedLanguage === option.value}
                    onClick={() => {
                      setSelectedLanguage(option.value as "auto" | WhisperLanguage);
                      setIsLangMenuOpen(false);
                    }}
                    className={[
                      "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors",
                      selectedLanguage === option.value
                        ? "bg-cyan-400/15 text-cyan-200"
                        : "text-neutral-200 hover:bg-neutral-800",
                    ].join(" ")}
                  >
                    <span className="text-base leading-none">{option.flag}</span>
                    {option.label}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>

        {/* Step 2 — Load AI model (hidden once ready) */}
        {!modelReady && (
          <div
            className={["mb-4", isModelShaking ? "model-shake" : ""].join(" ")}
            onAnimationEnd={() => setIsModelShaking(false)}
          >
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-neutral-400">
              Step 2 — Load AI model
            </p>
            {!selectedLanguage ? (
              <div className="flex items-center gap-2.5 rounded-xl border border-white/10 bg-neutral-900/80 px-3.5 py-2.5 opacity-50">
                <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor" className="shrink-0 text-neutral-400"><path d="M240-80q-33 0-56.5-23.5T160-160v-400q0-33 23.5-56.5T240-640h40v-80q0-83 58.5-141.5T480-920q83 0 141.5 58.5T680-720v80h40q33 0 56.5 23.5T800-560v400q0 33-23.5 56.5T720-80H240Zm0-80h480v-400H240v400Zm240-120q33 0 56.5-23.5T560-360q0-33-23.5-56.5T480-440q-33 0-56.5 23.5T400-360q0 33 23.5 56.5T480-280ZM480-640h160v-80q0-50-35-85t-85-35q-50 0-85 35t-35 85v80Zm-240 480v-400 400Z" /></svg>
                <span className="text-sm text-neutral-400">Select a language first</span>
              </div>
            ) : isCompiling ? (
              <div className="space-y-2 rounded-xl border border-violet-500/20 bg-violet-500/5 p-3">
                <p className="text-xs text-violet-300/90">First-time setup — compiling WebGPU shaders. Cached after this run.</p>
                <div className="flex items-center justify-between gap-4">
                  <div className="space-y-0.5">
                    <p className="text-xs text-neutral-300 sm:text-sm">Preparing GPU kernels… this takes 1–2 minutes on first run.</p>
                    <p className="text-xs text-neutral-500">You can leave this tab open and wait.</p>
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
                  <div className="h-full w-full animate-[shimmer_1.5s_ease-in-out_infinite] rounded-full bg-gradient-to-r from-violet-600/40 via-violet-400 to-violet-600/40 bg-[length:200%_100%]" />
                </div>
              </div>
            ) : progressPhase === "download" ? (
              <div className="space-y-2.5 rounded-xl border border-white/10 bg-neutral-950/70 p-3">
                <div className="flex items-center justify-between gap-4">
                  <p className="text-xs font-medium text-neutral-200 sm:text-sm" style={{ fontVariantNumeric: "tabular-nums" }}>{progressLabel}</p>
                  <button
                    type="button"
                    onClick={cancelTranscription}
                    disabled={isCancelling}
                    className="shrink-0 inline-flex items-center gap-1.5 rounded-md border border-red-500/40 bg-red-500/10 px-2.5 py-1 text-xs font-medium text-red-200 transition-colors hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <Square className="size-3.5" />
                    {isCancelling ? "Cancelling..." : "Cancel"}
                  </button>
                </div>
                <div className="h-2 overflow-hidden rounded-full border border-white/10 bg-neutral-900/90">
                  <div
                    className="h-full rounded-full bg-cyan-400 transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                {etaLabel ? (
                  <p className="text-xs text-neutral-500" style={{ fontVariantNumeric: "tabular-nums" }}>{etaLabel}</p>
                ) : null}
              </div>
            ) : (
              <div className="flex items-center gap-2.5 rounded-xl border border-white/10 bg-neutral-900/80 px-3.5 py-2.5">
                <svg className="size-4 animate-spin text-neutral-400" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm text-neutral-400">Loading model…</span>
              </div>
            )}
          </div>
        )}

        {/* Step 2/3 — Upload */}
        {uploadBusy ? (
          /* While processing: hide the full dropzone, show only the compact file row */
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-neutral-400">
              {modelReady ? "Step 2" : "Step 3"} — Upload your audio file
            </p>
            <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-neutral-900/80 px-3.5 py-2.5">
              <div className="flex min-w-0 items-center gap-2.5">
                <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor" className="shrink-0 text-neutral-400"><path d="M560-360v-240l80 80 56-56-160-160-160 160 56 56 80-80v240h48Zm-80 200q-83 0-141.5-58.5T280-360v-400h400v400q0 83-58.5 141.5T480-160Zm0-80q50 0 85-35t35-85v-320H360v320q0 50 35 85t85 35ZM200-80q-33 0-56.5-23.5T120-160v-520h80v520h520v80H200Zm280-440Z" /></svg>
                <span className="truncate text-sm text-neutral-200">{activeFileName}</span>
              </div>
              <span className="shrink-0 text-xs text-neutral-500">Processing…</span>
            </div>
          </div>
        ) : (
          <div className="relative">
            <div className={(!selectedLanguage || !modelReady) ? "pointer-events-none opacity-40" : ""}>
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-neutral-400">
                {modelReady ? "Step 2" : "Step 3"} — Upload your audio file
              </p>
              <UploadDropzone onFileSelected={handleFileSelected} />
            </div>
            {(!selectedLanguage || !modelReady) && (
              <div
                className="absolute inset-0 cursor-pointer"
                onClick={() => {
                  if (!selectedLanguage) { setIsLangShaking(true); setIsLangMenuOpen(true); }
                  if (!modelReady) { setIsModelShaking(true); }
                }}
              />
            )}
          </div>
        )}

        {isWarmingUp ? (
          <div className="mt-3 rounded-xl border border-cyan-500/20 bg-neutral-950/60 p-4 shadow-inner">
            {/* Header row */}
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-0.5">
                <p className="text-sm font-semibold text-neutral-100">
                  Transcription in progress
                </p>
                <p className="text-xs text-neutral-400">
                  Initial segment processing — the GPU is warming up. This takes 30–90 s the first time.
                </p>
              </div>
              <button
                type="button"
                onClick={cancelTranscription}
                disabled={isCancelling}
                className="shrink-0 inline-flex items-center gap-1.5 rounded-md border border-red-500/40 bg-red-500/10 px-2.5 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Square className="size-3.5" />
                {isCancelling ? "Cancelling…" : "Cancel"}
              </button>
            </div>

            {/* Stats row */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-md border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-1 text-xs font-medium tabular-nums text-cyan-300">
                <Clock3 className="size-3.5" style={{ animationDuration: "3s" }} />
                {warmUpElapsed}s elapsed
              </span>

              {totalSlices !== null && totalSlices > 1 && currentSlice !== null ? (
                <span className="inline-flex items-center rounded-md border border-white/10 bg-neutral-800/60 px-2.5 py-1 text-xs text-neutral-400">
                  Slice {currentSlice} / {totalSlices}
                </span>
              ) : null}

              {totalChunks !== null ? (
                <span className="inline-flex items-center rounded-md border border-white/10 bg-neutral-800/60 px-2.5 py-1 text-xs text-neutral-400">
                  0 / {totalChunks} segments
                </span>
              ) : null}

              {roughAudioMinutes !== null ? (
                <span className="inline-flex items-center rounded-md border border-white/10 bg-neutral-800/60 px-2.5 py-1 text-xs text-neutral-500">
                  ~{roughAudioMinutes} min audio
                </span>
              ) : null}
            </div>

            {/* Activity bar */}
            <div className="mt-3 h-1 overflow-hidden rounded-full bg-neutral-800">
              <div className="h-full w-full animate-[shimmer_1.5s_ease-in-out_infinite] rounded-full bg-gradient-to-r from-cyan-600/50 via-cyan-400 to-cyan-600/50 bg-[length:200%_100%]" />
            </div>

            <p className="mt-2 text-xs text-neutral-600">
              Keep this tab active while processing.
            </p>
          </div>
        ) : null}

        {showProgressBar ? (
          <div className="mt-3 space-y-2.5 rounded-xl border border-white/10 bg-neutral-950/70 p-3">
            {/* ── Top row: label + cancel ─────────────────────────────────── */}
            <div className="flex items-center justify-between gap-4">
              <p className="text-xs font-medium text-neutral-200 sm:text-sm" style={{ fontVariantNumeric: "tabular-nums" }}>{progressLabel}</p>
              <button
                type="button"
                onClick={cancelTranscription}
                disabled={isCancelling}
                className="shrink-0 inline-flex items-center gap-1.5 rounded-md border border-red-500/40 bg-red-500/10 px-2.5 py-1 text-xs font-medium text-red-200 transition-colors hover:bg-red-500/20 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Square className="size-3.5" />
                {isCancelling ? "Cancelling..." : "Cancel"}
              </button>
            </div>

            {/* ── Progress bar ────────────────────────────────────────────── */}
            <div className="h-2 overflow-hidden rounded-full border border-white/10 bg-neutral-900/90">
              <div
                className="h-full rounded-full bg-cyan-400 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* ── Slice pip track (only when there are multiple slices) ───── */}
            {totalSlices !== null && totalSlices > 1 ? (
              <div className="flex items-center gap-1">
                {Array.from({ length: totalSlices }).map((_, i) => (
                  <div
                    key={i}
                    className={[
                      "h-1 flex-1 rounded-full transition-colors duration-300",
                      currentSlice !== null && i < currentSlice
                        ? "bg-cyan-400"
                        : currentSlice !== null && i === currentSlice - 1
                          ? "bg-cyan-400/60"
                          : "bg-neutral-700",
                    ].join(" ")}
                  />
                ))}
              </div>
            ) : null}

            {/* ── ETA row ─────────────────────────────────────────────────── */}
            {etaLabel ? (
              <p className="text-xs text-neutral-500" style={{ fontVariantNumeric: "tabular-nums" }}>{etaLabel}</p>
            ) : null}
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

        <div className={["mt-4 overflow-hidden rounded-xl border border-white/10 bg-neutral-950/75", justCompleted ? "transcript-flash" : ""].join(" ")}>
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-4 py-2">
            <div className="flex items-center gap-3">
              <p className="text-sm font-medium text-neutral-200">Transcript Output</p>
              {output ? (
                <span className="text-xs text-neutral-500">
                  {output.trim().split(/\s+/).filter(Boolean).length} words
                </span>
              ) : null}
              {segments.length > 0 ? (
                <div className="flex items-center rounded-md border border-white/10 bg-neutral-900 p-0.5">
                  <button
                    type="button"
                    onClick={() => setViewMode("plain")}
                    className={[
                      "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                      viewMode === "plain"
                        ? "bg-cyan-400/15 text-cyan-200"
                        : "text-neutral-400 hover:text-neutral-200",
                    ].join(" ")}
                  >
                    Plain text
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode("timestamps")}
                    className={[
                      "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                      viewMode === "timestamps"
                        ? "bg-cyan-400/15 text-cyan-200"
                        : "text-neutral-400 hover:text-neutral-200",
                    ].join(" ")}
                  >
                    With timestamps
                  </button>
                </div>
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
            ) : segments.length > 0 && viewMode === "timestamps" ? (
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
                        className="text-sm leading-6 text-neutral-200 font-sans"
                      >
                        {segment.text}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
            ) : output ? (
              <textarea
                ref={outputTextareaRef}
                readOnly
                value={output}
                className={[
                  "w-full min-h-[220px] resize-none rounded-lg border border-white/10 bg-neutral-900/60 p-4 text-sm leading-6 text-neutral-200 outline-none font-sans",
                ].join(" ")}
              />
            ) : null}
          </div>
        </div>

        {output.trim() && status !== "transcribing" && status !== "decoding" && status !== "loading" ? (
          <div className="mt-3 rounded-xl border border-white/10 bg-neutral-900/50 px-4 py-3.5">
            <p className="mb-0.5 text-sm font-medium text-neutral-200">Continue with AI</p>
            <p className="mb-3 text-xs leading-relaxed text-neutral-500">
              Your transcript may be long. Open it in an AI chat to summarize, extract key points, ask questions, or generate study notes — the transcript is copied to clipboard automatically.
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(output.trim()).catch(() => { });
                  window.open("https://chatgpt.com/", "_blank", "noopener,noreferrer");
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-white/25 hover:bg-neutral-800 hover:text-white"
              >
                <svg height="13" viewBox="0 0 41 41" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M37.532 16.87a9.963 9.963 0 0 0-.856-8.184 10.078 10.078 0 0 0-10.855-4.835 9.964 9.964 0 0 0-6.99-3.118 10.079 10.079 0 0 0-9.613 6.977 9.967 9.967 0 0 0-6.664 4.834 10.08 10.08 0 0 0 1.24 11.817 9.965 9.965 0 0 0 .856 8.185 10.079 10.079 0 0 0 10.855 4.835 9.965 9.965 0 0 0 6.99 3.117 10.078 10.078 0 0 0 9.617-6.981 9.967 9.967 0 0 0 6.663-4.834 10.079 10.079 0 0 0-1.243-11.813zM22.498 37.886a7.474 7.474 0 0 1-4.799-1.735c.061-.033.168-.091.237-.134l7.964-4.6a1.294 1.294 0 0 0 .655-1.134V19.054l3.366 1.944a.12.12 0 0 1 .066.092v9.299a7.505 7.505 0 0 1-7.49 7.496zM6.392 31.006a7.471 7.471 0 0 1-.894-5.023c.06.036.162.099.237.141l7.964 4.6a1.297 1.297 0 0 0 1.308 0l9.724-5.614v3.888a.12.12 0 0 1-.048.103L16.552 34.1a7.504 7.504 0 0 1-10.16-3.094zM4.297 13.62a7.469 7.469 0 0 1 3.904-3.286c0 .068-.004.19-.004.274v9.201a1.294 1.294 0 0 0 .654 1.132l9.723 5.614-3.366 1.944a.12.12 0 0 1-.114.012L7.044 25.3a7.504 7.504 0 0 1-2.747-11.68zm23.232 6.386l-9.724-5.615 3.367-1.943a.121.121 0 0 1 .114-.012l8.048 4.648a7.498 7.498 0 0 1-1.158 13.528v-9.476a1.293 1.293 0 0 0-.647-1.13zm3.35-5.043c-.059-.037-.162-.099-.236-.141l-7.965-4.6a1.298 1.298 0 0 0-1.308 0l-9.723 5.614v-3.888a.12.12 0 0 1 .048-.103l8.031-4.637a7.5 7.5 0 0 1 11.153 7.755zm-21.063 6.929l-3.367-1.944a.12.12 0 0 1-.065-.092v-9.299a7.5 7.5 0 0 1 12.293-5.756 6.94 6.94 0 0 0-.236.134l-7.965 4.6a1.294 1.294 0 0 0-.654 1.132l-.006 11.225zm1.829-3.943l4.33-2.501 4.332 2.498v4.998l-4.331 2.5-4.331-2.5V19.386z" fill="currentColor" /></svg>
                ChatGPT
              </button>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(output.trim()).catch(() => { });
                  window.open("https://gemini.google.com/", "_blank", "noopener,noreferrer");
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-white/25 hover:bg-neutral-800 hover:text-white"
              >
                <svg height="13" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M14 28C14 26.0633 13.6267 24.2433 12.88 22.54C12.1567 20.8367 11.165 19.355 9.905 18.095C8.645 16.835 7.16333 15.8433 5.46 15.12C3.75667 14.3733 1.93667 14 0 14C1.93667 14 3.75667 13.6383 5.46 12.915C7.16333 12.1683 8.645 11.165 9.905 9.905C11.165 8.645 12.1567 7.16333 12.88 5.46C13.6267 3.75667 14 1.93667 14 0C14 1.93667 14.3617 3.75667 15.085 5.46C15.8317 7.16333 16.835 8.645 18.095 9.905C19.355 11.165 20.8367 12.1683 22.54 12.915C24.2433 13.6383 26.0633 14 28 14C26.0633 14 24.2433 14.3733 22.54 15.12C20.8367 15.8433 19.355 16.835 18.095 18.095C16.835 19.355 15.8317 20.8367 15.085 22.54C14.3617 24.2433 14 26.0633 14 28Z" fill="currentColor" /></svg>
                Gemini
              </button>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(output.trim()).catch(() => { });
                  window.open("https://grok.com/", "_blank", "noopener,noreferrer");
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-white/25 hover:bg-neutral-800 hover:text-white"
              >
                <svg height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M21.6 0H2.4C1.08 0 0 1.08 0 2.4v19.2C0 22.92 1.08 24 2.4 24h19.2c1.32 0 2.4-1.08 2.4-2.4V2.4C24 1.08 22.92 0 21.6 0zm-3.12 18.48h-2.04l-3.12-4.56-3.36 4.56H7.92l4.32-5.76L7.8 5.52h2.04l2.88 4.2 3.12-4.2h2.04l-4.2 5.52 4.8 7.44z" fill="currentColor" /></svg>
                Grok
              </button>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(output.trim()).catch(() => { });
                  window.open("https://copilot.microsoft.com/", "_blank", "noopener,noreferrer");
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-white/25 hover:bg-neutral-800 hover:text-white"
              >
                <svg height="13" viewBox="0 0 21 21" fill="none" xmlns="http://www.w3.org/2000/svg" className="shrink-0"><path d="M0 0h10v10H0V0zm11 0h10v10H11V0zM0 11h10v10H0V11zm11 0h10v10H11V11z" fill="currentColor" /></svg>
                Copilot
              </button>
            </div>
          </div>
        ) : null}

        <div className="mt-4 rounded-lg border border-white/10 bg-neutral-950/70 px-3 py-2 text-xs text-neutral-400 sm:text-sm">
          Supports <span className="font-medium text-neutral-300">.mp3, .wav, .m4a, .mp4, .ogg, .flac, .aac, .webm, .opus</span>.
          Transcription runs in-browser with{" "}
          <span className="font-medium text-neutral-300">Whisper Small</span>
          {isMobile ? <span> (on <span className="font-medium text-neutral-300">Groq API Whisper Large V3</span> via mobile fallback).</span> : "."}
          <br />
          <span className="mt-1 block text-amber-500/80">
            Do not refresh the page during transcription, or your progress will be lost.
          </span>
        </div>

        {busy ? (
          <p className="mt-2 text-xs text-neutral-500">
            Leave this tab open while processing long lectures for the best result stability.
          </p>
        ) : null}

        <div className="mt-6 flex items-center justify-between border-t border-white/10 pt-5">
          <p className="text-sm text-neutral-500">
            Developed by{" "}
            <span className="font-medium text-neutral-300">Onat Özmen</span>
          </p>
          <div className="flex items-center gap-3">
            <a
              href="/privacy"
              className="text-sm text-neutral-500 transition-colors hover:text-neutral-300"
            >
              Privacy Policy
            </a>
            <span className="text-neutral-700">·</span>
            <a
              href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-neutral-900 px-3 py-2 text-sm font-medium text-neutral-300 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800 hover:text-cyan-200"
            >
              <Linkedin className="size-4" />
              LinkedIn
            </a>
          </div>
        </div>
      </section>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "WebApplication",
            "name": "Audio Transcription Tool",
            "description": "Free, private, and 100% client-side audio transcription tool. Transcribe audio directly in your browser.",
            "applicationCategory": "MultimediaApplication",
            "operatingSystem": "Any",
            "offers": {
              "@type": "Offer",
              "price": "0",
              "priceCurrency": "USD"
            },
            "featureList": [
              "100% Client-side processing on Desktop",
              "Blazing fast Cloud processing on Mobile",
              "Privacy first - minimal data processing",
              "Supports multiple audio formats",
              "High accuracy with Whisper Small & Large V3",
              "Free to use"
            ]
          })
        }}
      />
    </main>
  );
}
