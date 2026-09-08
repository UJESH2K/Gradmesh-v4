import { currentUser } from "@/lib/auth";

import TrainingView from "./TrainingView";

export const metadata = { title: "Training" };
export const dynamic = "force-dynamic";

export default async function TrainingPage() {
  const user = await currentUser();
  return <TrainingView canManage={user?.role === "owner"} />;
}
