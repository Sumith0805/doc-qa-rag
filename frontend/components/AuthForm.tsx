"use client";
// One form used by both /login and /register. The `mode` prop decides the wording
// and which API call runs. "use client" means this runs in the browser, which it must,
// because it reacts to typing and clicks.

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ApiError, login, register } from "@/lib/api";

const TEXT = {
  login: {
    title: "Welcome back",
    subtitle: "Log in to your document library.",
    button: "Log in",
    busy: "Logging in...",
    other: { prompt: "New here?", label: "Create an account", href: "/register" },
  },
  register: {
    title: "Create your account",
    subtitle: "Your documents stay private to your account.",
    button: "Create account",
    busy: "Creating account...",
    other: { prompt: "Already have an account?", label: "Log in", href: "/login" },
  },
} as const;

export default function AuthForm({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const text = TEXT[mode];
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault(); // stop the browser's default full-page form submit
    setError("");
    setBusy(true);
    try {
      await (mode === "login" ? login(email, password) : register(email, password));
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setBusy(false);
    }
  }

  return (
    <main className="center-screen">
      <Image src="/logo-mark.png" alt="DocuMind" width={96} height={80} priority />
      <div className="card">
        <h1>{text.title}</h1>
        <p className="muted">{text.subtitle}</p>

        <form className="form" onSubmit={onSubmit}>
          <label className="field">
            Email
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label className="field">
            Password
            <input
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>

          {/* role="alert" makes screen readers announce the error when it appears */}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}

          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? text.busy : text.button}
          </button>
        </form>

        <p className="form-footer">
          {text.other.prompt} <Link href={text.other.href}>{text.other.label}</Link>
        </p>
      </div>
    </main>
  );
}
