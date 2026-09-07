import { currentUser } from "@/lib/auth";

import DatasetManager from "./DatasetManager";

export const metadata = { title: "Datasets" };
export const dynamic = "force-dynamic";

export default async function DatasetsPage() {
  const user = await currentUser();
  return <DatasetManager canManage={user?.role === "owner"} />;
}
