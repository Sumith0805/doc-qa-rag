// The ONE place the frontend talks to the FastAPI backend.
// Pages call the functions at the bottom; they never call fetch() themselves.

// NEXT_PUBLIC_ variables are copied into the browser bundle at build time, so they are
// PUBLIC. Only the API's address goes here. Never put a key or password in one.
const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

const TOKEN_KEY = "documind_token";

export type User = { id: string; email: string };

// ---------- token storage ----------
// The login token lives in localStorage. Trade-off: it survives page reloads and works when
// the site and the API are on different domains, but any script running on our page could
// read it. So: no third-party scripts, and never render HTML that came from a document.

export function getToken(): string | null {
  if (typeof window === "undefined") return null; // on the server there is no browser storage
  return window.localStorage.getItem(TOKEN_KEY);
}

function setToken(token: string) {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  window.localStorage.removeItem(TOKEN_KEY);
}

// ---------- requests ----------

/** An error the UI can show directly: `message` is the API's own `detail` text. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: { method?: string; body?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (options.body !== undefined) headers["Content-Type"] = "application/json";

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: options.method ?? "GET",
      headers,
      body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    });
  } catch {
    // fetch only throws when the server could not be reached at all
    throw new ApiError(0, "Cannot reach the server. Check your connection and try again.");
  }

  if (response.status === 204) return undefined as T; // "no content"

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : "Something went wrong.";
    if (response.status === 401 && token) clearToken(); // the saved token expired or was revoked
    throw new ApiError(response.status, detail);
  }
  return data as T;
}

// ---------- account calls ----------

export async function login(email: string, password: string): Promise<User> {
  const data = await request<{ token: string; user: User }>("/auth/login", {
    method: "POST",
    body: { email, password },
  });
  setToken(data.token);
  return data.user;
}

export async function register(email: string, password: string): Promise<User> {
  await request<{ user: User }>("/auth/register", { method: "POST", body: { email, password } });
  return login(email, password); // registering does not log you in, so do that next
}

export async function me(): Promise<User> {
  const data = await request<{ user: User }>("/auth/me");
  return data.user;
}

export async function logout(): Promise<void> {
  try {
    await request("/auth/logout", { method: "POST" }); // tells the server to forget the token
  } finally {
    clearToken(); // forget it in the browser even if the server call failed
  }
}
