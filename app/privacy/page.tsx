import type { Metadata } from "next";
import Link from "next/link";
import { ShieldCheck } from "lucide-react";

export const metadata: Metadata = {
    title: "Privacy Policy — Audio Transcription",
    description: "Privacy policy for the client-side audio transcription tool.",
};

const Section = ({
    title,
    children,
}: {
    title: string;
    children: React.ReactNode;
}) => (
    <div className="space-y-3">
        <h2 className="text-base font-semibold text-neutral-100">{title}</h2>
        <div className="space-y-2 text-sm leading-7 text-neutral-400">{children}</div>
    </div>
);

export default function PrivacyPolicy() {
    const updated = "February 21, 2026";

    return (
        <main className="flex min-h-screen items-start justify-center px-4 py-12 sm:px-6">
            <article className="w-full max-w-2xl rounded-2xl border border-white/10 bg-neutral-900/70 p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_24px_80px_rgba(0,0,0,0.55)] backdrop-blur-sm sm:p-10">

                {/* Back link */}
                <Link
                    href="/"
                    className="mb-8 inline-flex items-center gap-1.5 text-xs text-neutral-500 transition-colors hover:text-neutral-300"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" height="14px" viewBox="0 -960 960 960" width="14px" fill="currentColor">
                        <path d="M400-80 0-480l400-400 71 71-329 329 329 329-71 71Z" />
                    </svg>
                    Back to app
                </Link>

                {/* Header */}
                <header className="mb-8 space-y-3">
                    <p className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">
                        <ShieldCheck className="size-3.5" />
                        Privacy-first by design
                    </p>
                    <h1 className="text-2xl font-semibold tracking-tight text-white">
                        Privacy Policy
                    </h1>
                    <p className="text-xs text-neutral-500">
                        Last updated: {updated}
                    </p>
                </header>

                {/* Sections */}
                <div className="divide-y divide-white/5 space-y-8">

                    <div className="pt-8 space-y-3">
                        <Section title="Privacy Overview">
                            <p>
                                This tool is designed with a strong commitment to user privacy.
                                <span className="font-medium text-neutral-200">We do not collect any personal data.</span>
                                All audio processing for transcription happens either directly in your browser (desktop) or via a
                                secure, ephemeral API (mobile) that does not store your audio.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Overview">
                            <p>
                                This tool is built with privacy as a core principle. How your data is handled depends on the device you are using:
                            </p>
                            <ul className="list-disc space-y-2 pl-5">
                                <li>
                                    <span className="font-medium text-neutral-200">On Desktop:</span> All transcription happens entirely inside your browser. No audio data, text, or any other personal information is ever sent to a server.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">On Mobile:</span> Due to hardware limitations on phones, your audio is securely sent to <a href="https://groq.com" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300">Groq Cloud</a> for lightning-fast Processing. Groq processes this ephemerally and does not use your data to train their models.
                                </li>
                            </ul>
                        </Section>

                        <Section title="How Transcription Works">
                            <p>
                                <span className="font-medium text-neutral-200">Desktop Processing:</span> When you upload audio on a desktop, it is decoded and processed entirely within your browser using the Whisper Small model running via WebAssembly or WebGPU. The AI model itself is downloaded from Hugging Face once and cached in your browser — after that, the tool works fully offline.
                            </p>
                            <p className="mt-2">
                                <span className="font-medium text-neutral-200">Mobile Processing:</span> To prevent memory crashes on mobile browsers, your audio is chunked and securely sent via our API to <span className="font-medium text-neutral-200">Groq Cloud</span>, which uses the Whisper Large V3 model. The audio is transcribed and the text is immediately returned to your device. Groq operates under strict privacy guidelines and your audio is purely processed in-memory.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Browser Cache & Local Storage">
                            <p>
                                The Whisper model weights (approximately 150 MB) are cached in your browser&apos;s
                                native cache after the first download. This is standard browser caching behavior
                                (no different from a website caching its CSS or images). You can clear this cache
                                at any time through your browser settings.
                            </p>
                            <p>
                                We do not use <code className="rounded bg-neutral-800 px-1 py-0.5 text-xs font-mono text-neutral-300">localStorage</code>,{" "}
                                <code className="rounded bg-neutral-800 px-1 py-0.5 text-xs font-mono text-neutral-300">sessionStorage</code>, or{" "}
                                <code className="rounded bg-neutral-800 px-1 py-0.5 text-xs font-mono text-neutral-300">IndexedDB</code> for any purpose.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Third-Party Services">
                            <p>
                                Depending on your device, this application interacts with the following strict-privacy third parties:
                            </p>
                            <ul className="list-disc space-y-2 pl-5 mt-2">
                                <li>
                                    <span className="font-medium text-neutral-200">Hugging Face (Desktop):</span> Used only to download the open-source Whisper model files on first use. Their <a href="https://huggingface.co/privacy" target="_blank" rel="noopener noreferrer" className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300">privacy policy</a> applies to this simple file download.
                                </li>
                                <li>
                                    <span className="font-medium text-neutral-200">Groq Cloud (Mobile):</span> Used as the processing engine for mobile transcriptions. Groq processes API requests ephemerally; they do not retain your audio data or use it to train their models. You can read the <a href="https://groq.com/privacy-policy/" target="_blank" rel="noopener noreferrer" className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300">Groq Privacy Policy</a> for more details.
                                </li>
                            </ul>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Children's Privacy">
                            <p>
                                This service does not knowingly collect any data from children or anyone
                                else, as no data is collected at all.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Changes to This Policy">
                            <p>
                                If this policy ever changes, the updated version will be published at this
                                URL with a revised &quot;Last updated&quot; date.
                            </p>
                        </Section>
                    </div>

                    <div className="pt-8 space-y-3">
                        <Section title="Contact">
                            <p>
                                Questions? Reach out via{" "}
                                <a
                                    href="https://www.linkedin.com/in/onat-%C3%B6zmen-5b2212250"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-cyan-400 underline underline-offset-2 hover:text-cyan-300"
                                >
                                    LinkedIn
                                </a>
                                .
                            </p>
                        </Section>
                    </div>

                </div>

                {/* Footer */}
                <div className="mt-10 flex items-center justify-between border-t border-white/10 pt-6">
                    <p className="text-xs text-neutral-500">
                        Developed by{" "}
                        <span className="font-medium text-neutral-300">Onat Özmen</span>
                    </p>
                    <Link
                        href="/"
                        className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-neutral-900 px-2.5 py-1.5 text-xs font-medium text-neutral-300 transition-colors hover:border-cyan-400/40 hover:bg-neutral-800 hover:text-cyan-200"
                    >
                        ← Back to app
                    </Link>
                </div>

            </article>
        </main>
    );
}
