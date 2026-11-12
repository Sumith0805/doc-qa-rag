"use client";
// The intro screen at "/". It shows the logo briefly, then sends the visitor on:
// to /dashboard if a login token is saved, otherwise to /login.
// The button lets people skip the wait.

import { useEffect } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { getToken } from "@/lib/api";

const INTRO_MS = 1800;

export default function IntroPage() {
  const router = useRouter();

  function enter() {
    router.replace(getToken() ? "/dashboard" : "/login");
  }

  useEffect(() => {
    const timer = setTimeout(() => router.replace(getToken() ? "/dashboard" : "/login"), INTRO_MS);
    return () => clearTimeout(timer); // cancel the timer if the visitor leaves first
  }, [router]);

  return (
    <main className="center-screen">
      <Image
        className="intro-logo"
        src="/logo-full.png"
        alt="DocuMind"
        width={745}
        height={520}
        priority
      />
      <p className="intro-tagline">Answers from your documents, with the page to prove it.</p>
      <button className="btn btn-primary" onClick={enter}>
        Continue
      </button>
    </main>
  );
}
