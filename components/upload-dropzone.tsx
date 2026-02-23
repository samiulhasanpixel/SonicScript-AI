"use client";

import { UploadCloud, XCircle } from "lucide-react";
import { useRef, useState } from "react";

export type AcceptedAudioExtension = "mp3" | "wav" | "m4a" | "mp4" | "ogg" | "flac" | "aac" | "webm" | "opus";

export interface UploadDropzoneProps {
  onFileSelected?: (file: File) => void;
}

const ACCEPTED_AUDIO_EXTENSIONS: AcceptedAudioExtension[] = ["mp3", "wav", "m4a", "mp4", "ogg", "flac", "aac", "webm", "opus"];
const ACCEPT_ATTRIBUTE =
  ".mp3,.wav,.m4a,.mp4,.ogg,.flac,.aac,.webm,.opus,audio/mpeg,audio/wav,audio/x-wav,audio/mp4,audio/x-m4a,audio/ogg,audio/flac,audio/aac,audio/webm,audio/opus";

export function isAcceptedAudioFile(file: File): boolean {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return (
    extension !== undefined &&
    ACCEPTED_AUDIO_EXTENSIONS.includes(extension as AcceptedAudioExtension)
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadDropzone({ onFileSelected }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (file: File | null) => {
    if (!file) return;

    if (!isAcceptedAudioFile(file)) {
      setSelectedFile(null);
      setError("Unsupported file type. Please upload .mp3, .wav, .m4a, .mp4, .ogg, .flac, .aac, .webm, or .opus.");
      return;
    }

    setError(null);
    setSelectedFile(file);
    onFileSelected?.(file);
  };

  const openFilePicker = () => {
    inputRef.current?.click();
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0] ?? null;
          handleFile(file);
        }}
      />

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an audio file"
        onClick={openFilePicker}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openFilePicker();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          const nextTarget = event.relatedTarget;
          if (nextTarget && event.currentTarget.contains(nextTarget as Node)) {
            return;
          }
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          const file = event.dataTransfer.files?.[0] ?? null;
          handleFile(file);
        }}
        className={[
          "group relative flex min-h-64 w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border border-dashed p-8 text-center outline-none transition-all duration-200 ease-out",
          "hover:border-neutral-500/80 hover:bg-neutral-800/60 focus-visible:ring-2 focus-visible:ring-cyan-400/60",
          isDragging
            ? "scale-[1.01] border-cyan-400/80 bg-cyan-400/10 shadow-[0_0_0_1px_rgba(34,211,238,0.25),0_0_40px_rgba(34,211,238,0.15)]"
            : "border-neutral-700 bg-neutral-900/60",
          !isDragging && error ? "border-red-500/70 bg-red-500/10" : "",
        ].join(" ")}
      >
        <div
          className={[
            "rounded-full border p-3 transition-colors",
            isDragging
              ? "border-cyan-300/70 bg-cyan-300/15 text-cyan-200"
              : "border-neutral-700 bg-neutral-800 text-neutral-300 group-hover:border-neutral-500",
          ].join(" ")}
        >
          {error ? <XCircle className="size-6" /> : <UploadCloud className="size-6" />}
        </div>

        <div className="space-y-2">
          <p className="text-base font-medium text-neutral-100">
            {isDragging ? "Drop your audio file here" : "Drag and drop audio to upload"}
          </p>
          <p className="text-sm text-neutral-400">
            or <span className="text-cyan-300">browse from your device</span>
          </p>
        </div>

        <p className="text-xs text-neutral-500">Accepted formats: .mp3, .wav, .m4a, .mp4, .ogg, .flac, .aac, .webm, .opus</p>
      </div>

      {error ? (
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          <XCircle className="size-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      {selectedFile ? (
        <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-neutral-900/80 px-3.5 py-2.5">
          <div className="flex min-w-0 items-center gap-2.5">
            <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor" className="shrink-0 text-neutral-400"><path d="M560-360v-240l80 80 56-56-160-160-160 160 56 56 80-80v240h48Zm-80 200q-83 0-141.5-58.5T280-360v-400h400v400q0 83-58.5 141.5T480-160Zm0-80q50 0 85-35t35-85v-320H360v320q0 50 35 85t85 35ZM200-80q-33 0-56.5-23.5T120-160v-520h80v520h520v80H200Zm280-440Z" /></svg>
            <span className="truncate text-sm text-neutral-200">{selectedFile.name}</span>
          </div>
          <span className="shrink-0 text-xs tabular-nums text-neutral-500">{formatFileSize(selectedFile.size)}</span>
        </div>
      ) : null}
    </div>
  );
}
