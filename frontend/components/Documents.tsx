"use client";
// The document library: upload a file (with progress), see each file's status, delete a file.

import { useEffect, useRef, useState } from "react";
import { ApiError, deleteDocument, listDocuments, uploadDocument, type Doc } from "@/lib/api";

// What the upload is doing right now. null = nothing in progress.
type Upload = { name: string; percent: number } | null;

function message(err: unknown) {
  return err instanceof ApiError ? err.message : "Something went wrong.";
}

export default function Documents() {
  const [docs, setDocs] = useState<Doc[] | null>(null); // null = still loading
  const [upload, setUpload] = useState<Upload>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch((err) => {
        setDocs([]);
        setError(message(err));
      });
  }, []);

  async function onPick(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = ""; // so picking the same file again still fires this handler
    if (!file) return;
    setError("");
    setNotice("");
    setUpload({ name: file.name, percent: 0 });
    try {
      const result = await uploadDocument(file, (percent) => setUpload({ name: file.name, percent }));
      setNotice(
        result.duplicate
          ? `"${file.name}" is already in your library.`
          : `"${file.name}" is ready. You can ask about it now.`,
      );
      setDocs(await listDocuments());
    } catch (err) {
      setError(message(err));
    } finally {
      setUpload(null);
    }
  }

  async function onDelete(doc: Doc) {
    if (!window.confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    setError("");
    setNotice("");
    try {
      await deleteDocument(doc.id);
      setDocs((current) => (current ?? []).filter((d) => d.id !== doc.id));
    } catch (err) {
      setError(message(err));
    }
  }

  return (
    <section className="pane-body" aria-label="Documents">
      <input ref={fileInput} type="file" accept=".pdf,.txt,.md" hidden onChange={onPick} />
      <button className="btn" onClick={() => fileInput.current?.click()} disabled={upload !== null}>
        Upload a document
      </button>
      <p className="hint">PDF, TXT or MD. This is a demo: do not upload private documents.</p>

      {upload && (
        <div className="upload" role="status">
          <div className="upload-name">{upload.name}</div>
          <div className="progress">
            <div className="progress-bar" style={{ width: `${upload.percent}%` }} />
          </div>
          <div className="hint">
            {upload.percent < 100 ? `Uploading ${upload.percent}%` : "Reading and indexing..."}
          </div>
        </div>
      )}

      {notice && (
        <p className="notice" role="status">
          {notice}
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {docs === null ? (
        <p className="muted">Loading your documents...</p>
      ) : docs.length === 0 ? (
        <p className="muted">No documents yet. Upload one to start asking questions.</p>
      ) : (
        <ul className="doc-list">
          {docs.map((doc) => (
            <li key={doc.id} className="doc">
              <div className="doc-main">
                <div className="doc-name" title={doc.filename}>
                  {doc.filename}
                </div>
                <div className="hint">
                  <span className={`status-${doc.status}`}>{doc.status}</span>
                  {doc.pages ? ` · ${doc.pages} ${doc.pages === 1 ? "page" : "pages"}` : ""}
                  {doc.status === "failed" && doc.error ? ` · ${doc.error}` : ""}
                </div>
              </div>
              <button
                className="btn btn-small"
                onClick={() => onDelete(doc)}
                aria-label={`Delete ${doc.filename}`}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
