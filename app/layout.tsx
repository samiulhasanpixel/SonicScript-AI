import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Audio Transcription Tool | Private & Local De-script",
    template: "%s | Audio Transcription Tool",
  },
  description: "Free, private, and 100% client-side audio transcription tool. Transcribe lectures, meetings, and voice notes directly in your browser without uploading any data to servers.",
  keywords: ["audio transcription", "speech to text", "private transcription", "local transcription", "whisper ai", "client-side transcription", "browser-based transcription", "free transcription tool"],
  authors: [{ name: "Audio Transcription Tool" }],
  creator: "Audio Transcription Tool",
  publisher: "Audio Transcription Tool",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    title: "Audio Transcription Tool | Private & Local",
    description: "Transcribe audio files securely in your browser. No server uploads, total privacy.",
    url: "https://audio-transcription-tool-dun.vercel.app",
    siteName: "Audio Transcription Tool",
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Audio Transcription Tool | Private & Local",
    description: "Transcribe audio files securely in your browser. No server uploads, total privacy.",
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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} bg-neutral-950 text-neutral-100 antialiased`}>
        {children}
      </body>
    </html>
  );
}
