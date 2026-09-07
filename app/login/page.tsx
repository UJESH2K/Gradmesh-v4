import { redirect } from "next/navigation";

import { currentUser, needsFirstAccount } from "@/lib/auth";
import { meshName } from "@/lib/config";

import AuthForm from "./AuthForm";
import "./auth.css";

export const metadata = { title: "Sign in" };
export const dynamic = "force-dynamic";

export default async function LoginPage() {
  if (await currentUser()) redirect("/dashboard");
  return <AuthForm firstAccount={needsFirstAccount()} meshName={meshName()} />;
}
