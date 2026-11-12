"use client";
// The dashboard at /dashboard. Only logged-in users get here: useUser() is the guard.
// Wide screens show Documents and Chat side by side. Phones show one at a time, with tabs.

import { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import Chat from "@/components/Chat";
import Documents from "@/components/Documents";
import { logout } from "@/lib/api";
import { useUser } from "@/lib/useUser";

type Tab = "chat" | "documents";

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading } = useUser();
  const [leaving, setLeaving] = useState(false);
  const [tab, setTab] = useState<Tab>("documents");

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
    <div className="app">
      <header className="topbar">
        <div className="topbar-brand">
          <Image src="/logo-mark.png" alt="" width={36} height={30} />
          DocuMind
        </div>
        <div className="topbar-user">
          <span className="topbar-email">{user.email}</span>
          <button className="btn btn-small" onClick={onLogout} disabled={leaving}>
            {leaving ? "Logging out..." : "Log out"}
          </button>
        </div>
      </header>

      {/* Phone-only tabs. aria-pressed tells screen readers which one is active. */}
      <nav className="tabs" aria-label="Sections">
        <button aria-pressed={tab === "documents"} onClick={() => setTab("documents")}>
          Documents
        </button>
        <button aria-pressed={tab === "chat"} onClick={() => setTab("chat")}>
          Chat
        </button>
      </nav>

      <main className="panes" data-tab={tab}>
        <div className="pane pane-documents">
          <h2>Documents</h2>
          <Documents />
        </div>
        <div className="pane pane-chat">
          <h2>Chat</h2>
          <Chat />
        </div>
      </main>
    </div>
  );
}
