// The root layout wraps EVERY page. It sets the <html> and <body> tags, loads the font
// and the global stylesheet, and defines the browser-tab title.
import type { Metadata, Viewport } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans", // exposed to CSS as var(--font-geist-sans)
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DocuMind",
  description: "Ask questions about your documents and get answers with page citations.",
};

export const viewport: Viewport = {
  themeColor: "#08080c",
  viewportFit: "cover", // lets the page use the full iPhone screen; CSS handles the safe areas
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={geistSans.variable}>
      <body>{children}</body>
    </html>
  );
}
