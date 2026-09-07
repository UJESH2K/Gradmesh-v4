import "server-only";

import {
  createHmac,
  randomBytes,
  randomUUID,
  scryptSync,
  timingSafeEqual,
} from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { cookies } from "next/headers";

import { STATE_DIR, ensureStateDir, sessionSecret } from "./config";

/**
 * Dashboard accounts.
 *
 * Scope is deliberately narrow: this authenticates people looking at the mesh,
 * it is not the mesh's own trust boundary. Workers authenticate separately with
 * the mesh token, so a compromised dashboard password cannot make a machine
 * start receiving dataset shards.
 *
 * The first account created becomes the owner. Owners can start and stop runs,
 * change scheduling policy, upload datasets and evict nodes. Members get a
 * read-only view of the same data.
 */

export const SESSION_COOKIE = "gradmesh_session";
const SESSION_TTL_SECONDS = 60 * 60 * 24 * 14;
const USERS_FILE = () => path.join(STATE_DIR, "users.json");

export type Role = "owner" | "member";

export type User = {
  id: string;
  email: string;
  name: string;
  role: Role;
  passwordHash: string;
  salt: string;
  createdAt: number;
};

export type PublicUser = Omit<User, "passwordHash" | "salt">;

type UserFile = { users: User[] };

function readUsers(): UserFile {
  ensureStateDir();
  const file = USERS_FILE();
  if (!existsSync(file)) return { users: [] };
  try {
    const parsed = JSON.parse(readFileSync(file, "utf8"));
    return { users: Array.isArray(parsed.users) ? parsed.users : [] };
  } catch {
    return { users: [] };
  }
}

function writeUsers(data: UserFile): void {
  ensureStateDir();
  writeFileSync(USERS_FILE(), JSON.stringify(data, null, 2), { mode: 0o600 });
}

export function userCount(): number {
  return readUsers().users.length;
}

export function needsFirstAccount(): boolean {
  return userCount() === 0;
}

function hashPassword(password: string, salt: string): string {
  // scrypt with the Node defaults, which are deliberately expensive.
  return scryptSync(password, salt, 64).toString("hex");
}

function publicUser(user: User): PublicUser {
  const { passwordHash: _hash, salt: _salt, ...rest } = user;
  return rest;
}

export function createUser(input: {
  email: string;
  password: string;
  name?: string;
}): { ok: true; user: PublicUser } | { ok: false; error: string } {
  const email = input.email.trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    return { ok: false, error: "Enter a valid email address." };
  }
  if (input.password.length < 8) {
    return { ok: false, error: "Use a password of at least 8 characters." };
  }

  const data = readUsers();
  if (data.users.some((user) => user.email === email)) {
    return { ok: false, error: "An account with that email already exists." };
  }

  const salt = randomBytes(16).toString("hex");
  const user: User = {
    id: randomUUID(),
    email,
    name: input.name?.trim() || email.split("@")[0],
    // Whoever sets the mesh up owns it. Everyone after that joins as a member.
    role: data.users.length === 0 ? "owner" : "member",
    passwordHash: hashPassword(input.password, salt),
    salt,
    createdAt: Date.now(),
  };

  data.users.push(user);
  writeUsers(data);
  return { ok: true, user: publicUser(user) };
}

export function verifyCredentials(email: string, password: string): PublicUser | null {
  const user = readUsers().users.find((item) => item.email === email.trim().toLowerCase());
  if (!user) {
    // Spend the same work either way so a wrong email is not faster than a
    // wrong password.
    hashPassword(password, "decoy-salt-value");
    return null;
  }
  const candidate = Buffer.from(hashPassword(password, user.salt), "hex");
  const expected = Buffer.from(user.passwordHash, "hex");
  if (candidate.length !== expected.length || !timingSafeEqual(candidate, expected)) return null;
  return publicUser(user);
}

export function listUsers(): PublicUser[] {
  return readUsers().users.map(publicUser);
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

type SessionPayload = { sub: string; email: string; name: string; role: Role; exp: number };

function base64url(input: Buffer | string): string {
  return Buffer.from(input).toString("base64url");
}

function sign(payload: string): string {
  return createHmac("sha256", sessionSecret()).update(payload).digest("base64url");
}

export function createSessionToken(user: PublicUser): string {
  const payload: SessionPayload = {
    sub: user.id,
    email: user.email,
    name: user.name,
    role: user.role,
    exp: Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS,
  };
  const body = base64url(JSON.stringify(payload));
  return `${body}.${sign(body)}`;
}

export function readSessionToken(token: string | undefined): SessionPayload | null {
  if (!token || !token.includes(".")) return null;
  const [body, signature] = token.split(".");
  const expected = sign(body);
  if (
    signature.length !== expected.length ||
    !timingSafeEqual(Buffer.from(signature), Buffer.from(expected))
  ) {
    return null;
  }
  try {
    const payload = JSON.parse(Buffer.from(body, "base64url").toString("utf8")) as SessionPayload;
    if (payload.exp * 1000 < Date.now()) return null;
    return payload;
  } catch {
    return null;
  }
}

export async function currentUser(): Promise<SessionPayload | null> {
  const store = await cookies();
  return readSessionToken(store.get(SESSION_COOKIE)?.value);
}

export async function requireUser(): Promise<SessionPayload> {
  const user = await currentUser();
  if (!user) throw new Error("Not signed in");
  return user;
}

export async function requireOwner(): Promise<SessionPayload> {
  const user = await requireUser();
  if (user.role !== "owner") {
    throw new Error("Only the mesh owner can do that.");
  }
  return user;
}

export async function startSession(user: PublicUser): Promise<void> {
  const store = await cookies();
  store.set(SESSION_COOKIE, createSessionToken(user), {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_TTL_SECONDS,
    // The mesh runs over plain HTTP on a LAN, so a Secure cookie would never
    // be sent back. The trust boundary here is the network itself.
    secure: false,
  });
}

export async function endSession(): Promise<void> {
  const store = await cookies();
  store.delete(SESSION_COOKIE);
}
