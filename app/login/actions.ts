"use server";

import { redirect } from "next/navigation";

import {
  createUser,
  endSession,
  needsFirstAccount,
  startSession,
  verifyCredentials,
} from "@/lib/auth";

export type AuthState = { error: string | null };

export async function signIn(_previous: AuthState, formData: FormData): Promise<AuthState> {
  const email = String(formData.get("email") || "");
  const password = String(formData.get("password") || "");

  if (!email || !password) return { error: "Enter your email and password." };

  const user = verifyCredentials(email, password);
  if (!user) return { error: "That email and password do not match an account." };

  await startSession(user);
  redirect("/dashboard");
}

export async function signUp(_previous: AuthState, formData: FormData): Promise<AuthState> {
  const email = String(formData.get("email") || "");
  const password = String(formData.get("password") || "");
  const name = String(formData.get("name") || "");
  const first = needsFirstAccount();

  const result = createUser({ email, password, name });
  if (!result.ok) return { error: result.error };

  await startSession(result.user);
  redirect(first ? "/dashboard?welcome=1" : "/dashboard");
}

export async function signOut(): Promise<void> {
  await endSession();
  redirect("/");
}
