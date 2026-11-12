"use client";
// The auth guard. A page that needs a logged-in user calls useUser().
// While it checks, `user` is null and `loading` is true. If nobody is logged in
// (no token, or the server rejects it), the visitor is sent to /login.

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getToken, me, type User } from "./api";

export function useUser() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false; // ignore the answer if the page was left before it arrived

    async function check() {
      if (!getToken()) {
        router.replace("/login");
        return;
      }
      try {
        const current = await me(); // the SERVER decides whether the token is valid
        if (!cancelled) {
          setUser(current);
          setLoading(false);
        }
      } catch {
        if (!cancelled) router.replace("/login");
      }
    }

    check();
    return () => {
      cancelled = true;
    };
  }, [router]);

  return { user, loading };
}
