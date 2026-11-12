"use client";
// The dashboard at /dashboard. Only logged-in users get here: useUser() is the guard.
// Phase 3 adds documents, chat and history to this page.

import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { logout } from "@/lib/api";
import { useUser } from "@/lib/useUser";

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading } = useUser();
  const [leaving, setLeaving] = useState(false);

  async function onLogout() {
    setLeaving(true);
    try {
      await logout();
    } catch {
      // The token is cleared in the browser either way, so carry on to /login.
    }
    router.replace("/login");
  }

  if (loading || !user) {
    return (
      <main className="center-screen">
        <p className="muted">Loading...</p>
      </main>
    );
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-brand">
          <Image src="/logo-mark.png" alt="" width={36} height={30} />
          DocuMind
        </div>
        <div className="topbar-user">
          <span className="topbar-email">{user.email}</span>
          <button className="btn" onClick={onLogout} disabled={leaving}>
            {leaving ? "Logging out..." : "Log out"}
          </button>
        </div>
      </header>
      <main className="page">
        <h1>Your library</h1>
        <p className="muted">You are logged in. Documents and chat arrive in the next phase.</p>
      </main>
    </>
  );
}
