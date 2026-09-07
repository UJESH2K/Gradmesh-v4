import { headers } from "next/headers";

import Landing from "@/components/Landing";
import { currentUser, needsFirstAccount } from "@/lib/auth";
import { meshName, meshOrigin } from "@/lib/config";

import "./landing.css";

export default async function HomePage() {
  const headerList = await headers();
  const user = await currentUser();

  return (
    <Landing
      origin={meshOrigin(headerList.get("host"))}
      signedIn={Boolean(user)}
      needsFirstAccount={needsFirstAccount()}
      meshName={meshName()}
    />
  );
}
