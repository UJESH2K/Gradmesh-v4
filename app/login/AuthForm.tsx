"use client";

import { useActionState, useState } from "react";
import Link from "next/link";

import Logo from "@/components/Logo";
import { signIn, signUp, type AuthState } from "./actions";

const INITIAL: AuthState = { error: null };

export default function AuthForm({
  firstAccount,
  meshName,
}: {
  firstAccount: boolean;
  meshName: string;
}) {
  // When no account exists yet, whoever is at the keyboard is the mesh owner,
  // so there is nothing to sign in to and the form starts in create mode.
  const [mode, setMode] = useState<"signin" | "signup">(firstAccount ? "signup" : "signin");
  const action = mode === "signin" ? signIn : signUp;
  const [state, formAction, pending] = useActionState(action, INITIAL);

  return (
    <main className="auth-shell">
      <div className="auth-card">
        <Link href="/" className="brand" style={{ marginBottom: 26 }}>
          <Logo />
          <span>{meshName}</span>
        </Link>

        {firstAccount ? (
          <>
            <h2>Claim this mesh</h2>
            <p className="muted small" style={{ marginTop: 8 }}>
              No account exists yet. The first one becomes the owner: it can start runs, upload
              datasets, tune the scheduler and evict machines.
            </p>
          </>
        ) : (
          <>
            <h2>{mode === "signin" ? "Sign in" : "Create an account"}</h2>
            <p className="muted small" style={{ marginTop: 8 }}>
              {mode === "signin"
                ? "Dashboard access for this mesh."
                : "New accounts join as members and get a read-only view."}
            </p>
          </>
        )}

        <form action={formAction} className="stack" style={{ marginTop: 26 }}>
          {mode === "signup" ? (
            <div className="field">
              <label className="label" htmlFor="name">
                Name
              </label>
              <input className="input" id="name" name="name" autoComplete="name" placeholder="Ada" />
            </div>
          ) : null}

          <div className="field">
            <label className="label" htmlFor="email">
              Email
            </label>
            <input
              className="input"
              id="email"
              name="email"
              type="email"
              required
              autoComplete="email"
              placeholder="you@example.com"
            />
          </div>

          <div className="field">
            <label className="label" htmlFor="password">
              Password
            </label>
            <input
              className="input"
              id="password"
              name="password"
              type="password"
              required
              minLength={8}
              autoComplete={mode === "signin" ? "current-password" : "new-password"}
              placeholder="At least 8 characters"
            />
          </div>

          {state.error ? <div className="notice notice-danger">{state.error}</div> : null}

          <button className="btn btn-primary btn-lg" type="submit" disabled={pending}>
            {pending
              ? "Working…"
              : mode === "signin"
                ? "Sign in"
                : firstAccount
                  ? "Claim the mesh"
                  : "Create account"}
          </button>
        </form>

        {!firstAccount ? (
          <p className="small faint" style={{ marginTop: 20, textAlign: "center" }}>
            {mode === "signin" ? "No account yet?" : "Already have an account?"}{" "}
            <button
              type="button"
              className="btn-ghost"
              style={{
                background: "none",
                border: "none",
                padding: 0,
                color: "var(--accent)",
                cursor: "pointer",
              }}
              onClick={() => setMode(mode === "signin" ? "signup" : "signin")}
            >
              {mode === "signin" ? "Create one" : "Sign in"}
            </button>
          </p>
        ) : null}

        <p className="small faint" style={{ marginTop: 26, textAlign: "center" }}>
          Lending a GPU does not need an account.{" "}
          <Link href="/join" style={{ color: "var(--text-dim)", textDecoration: "underline" }}>
            Go to the join page
          </Link>
        </p>
      </div>
    </main>
  );
}
