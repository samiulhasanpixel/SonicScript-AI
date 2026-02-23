import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Free Audio Transcription Tool — No Sign-up, No Uploads, AI-Powered",
    template: "%s | Free Audio Transcription Tool",
  },
  description:
    "Transcribe audio files completely free — no sign-up, no uploads, no limits. Powered by Whisper AI and runs entirely in your browser. Supports MP3, WAV, M4A, MP4, OGG, FLAC and more. Desktop and mobile friendly.",
  keywords: [
    "free audio transcription",
    "free speech to text",
    "audio to text free",
    "transcribe audio online free",
    "whisper ai transcription",
    "ai transcription",
    "browser transcription",
    "offline transcription",
    "private transcription",
    "lecture transcription",
    "meeting transcription",
    "no signup transcription",
    "online audio transcription",
    "mp3 to text",
    "wav to text",
    "m4a to text",
    "audio transcription tool",
    "speech to text online",
    "free voice to text",
    "client-side transcription",
  ],
  authors: [{ name: "Onat Özmen" }],
  creator: "Onat Özmen",
  publisher: "Onat Özmen",
  alternates: {
    canonical: "https://audio-transcription-tool-dun.vercel.app",
  },
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    title: "Free Audio Transcription Tool — No Sign-up, AI-Powered",
    description:
      "Transcribe audio completely free. Whisper AI runs in your browser — no uploads, no accounts, total privacy. Works on desktop and mobile.",
    url: "https://audio-transcription-tool-dun.vercel.app",
    siteName: "Free Audio Transcription Tool",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Free Audio Transcription Tool — No Sign-up, AI-Powered",
    description:
      "Transcribe audio free in your browser. Whisper AI — no uploads, no tracking, no account needed.",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  name: "Free Audio Transcription Tool",
  url: "https://audio-transcription-tool-dun.vercel.app",
  description:
    "Completely free, private, and AI-powered audio transcription tool. Powered by Whisper AI and runs entirely in your browser — no uploads, no sign-up, no data collection.",
  applicationCategory: "MultimediaApplication",
  operatingSystem: "Web Browser",
  browserRequirements: "Requires a modern browser with WebAssembly support. WebGPU recommended for best performance.",
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
    description: "Completely free, no hidden costs or subscriptions.",
  },
  featureList: [
    "100% Free",
    "No account required",
    "No file uploads",
    "Private and secure",
    "Works offline after first load",
    "AI-powered by Whisper",
    "Mobile friendly",
    "Supports MP3, WAV, M4A, MP4, OGG, FLAC, AAC, WEBM, OPUS",
  ],
  creator: {
    "@type": "Person",
    name: "Onat Özmen",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className={`${inter.variable} bg-neutral-950 text-neutral-100 antialiased`}>
        {children}
      </body>
    </html>
  );
}
