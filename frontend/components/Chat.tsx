"use client";
// The chat: past questions (history) plus new ones. Each answer shows its citations as
// buttons; pressing one reveals the exact passage. When the documents do not support an
// answer, the reply is shown in a clearly different "not found" style.

import { useEffect, useRef, useState } from "react";
import { ApiError, ask, clearHistory, getHistory, type Citation } from "@/lib/api";

type Turn = {
  key: string;
  question: string;
  // Until the reply arrives, both `answer` and `error` are undefined.
  answer?: string;
  error?: string;
  abstained?: boolean;
  citations?: Citation[];
  warning?: string | null;
};

function message(err: unknown) {
  return err instanceof ApiError ? err.message : "Something went wrong.";
}

function Reply({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState<string | null>(null); // label of the citation being shown
  const shown = turn.citations?.find((c) => c.label === open);

  if (turn.error) {
    return (
      <p className="error" role="alert">
        {turn.error}
      </p>
    );
  }
  if (turn.answer === undefined) {
    return <p className="bubble bubble-answer muted">Searching your documents...</p>;
  }
  if (turn.abstained) {
    return (
      <div className="bubble bubble-notfound">
        <div className="notfound-label">No answer from your documents</div>
        {turn.answer}
      </div>
    );
  }
  return (
    <div className="bubble bubble-answer">
      <div className="answer-text">{turn.answer}</div>
      {turn.warning && <div className="hint">{turn.warning}</div>}
      {turn.citations && turn.citations.length > 0 && (
        <div className="citations">
          {turn.citations.map((c) => (
            <button
              key={c.label}
              className="citation"
              aria-expanded={open === c.label}
              onClick={() => setOpen(open === c.label ? null : c.label)}
            >
              [{c.label}] {c.filename}
              {c.page ? `, page ${c.page}` : ""}
            </button>
          ))}
        </div>
      )}
      {shown && (
        <blockquote className="passage">
          {/* Rendered as plain text on purpose: document content is never treated as HTML. */}
          {shown.text}
        </blockquote>
      )}
    </div>
  );
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHistory()
      .then((messages) =>
        setTurns(
          messages.map((m) => ({
            key: `h${m.id}`,
            question: m.question,
            answer: m.answer,
            abstained: m.abstained,
            citations: m.sources,
          })),
        ),
      )
      .catch((err) => setError(message(err)))
      .finally(() => setLoaded(true));
  }, []);

  // Keep the newest message in view.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    const key = `n${Date.now()}`;
    setError("");
    setQuestion("");
    setBusy(true);
    setTurns((current) => [...current, { key, question: text }]);

    let update: Partial<Turn>;
    try {
      const result = await ask(text);
      update = {
        answer: result.answer,
        abstained: result.abstained,
        citations: result.citations,
        warning: result.warning,
      };
    } catch (err) {
      // 429 (too many questions) and 503 (AI service busy) arrive here with a readable message.
      update = { error: message(err) };
    }
    setTurns((current) => current.map((t) => (t.key === key ? { ...t, ...update } : t)));
    setBusy(false);
  }

  async function onClear() {
    if (!window.confirm("Clear your whole chat history?")) return;
    try {
      await clearHistory();
      setTurns([]);
    } catch (err) {
      setError(message(err));
    }
  }

  return (
    <section className="chat" aria-label="Chat">
      <div className="chat-scroll">
        {loaded && turns.length === 0 && (
          <p className="muted chat-empty">
            Ask a specific question, such as a name, date or requirement in your documents. Answers cite the file and page they came from.
          </p>
        )}
        {turns.map((turn) => (
          <div key={turn.key} className="turn">
            <p className="bubble bubble-question">{turn.question}</p>
            <Reply turn={turn} />
          </div>
        ))}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <div ref={bottom} />
      </div>

      <form className="composer" onSubmit={onSubmit}>
        <input
          aria-label="Your question"
          placeholder="Ask a specific question about your documents"
          maxLength={1000}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button className="btn btn-primary" type="submit" disabled={busy || !question.trim()}>
          {busy ? "Asking..." : "Ask"}
        </button>
      </form>
      {turns.length > 0 && (
        <button className="link-button" onClick={onClear}>
          Clear history
        </button>
      )}
    </section>
  );
}
