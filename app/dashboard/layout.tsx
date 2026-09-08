import { redirect } from "next/navigation";

import { MeshProvider } from "@/components/dashboard/MeshProvider";
import Sidebar from "@/components/dashboard/Sidebar";
import SetupBanner from "@/components/dashboard/SetupBanner";
import Topbar from "@/components/dashboard/Topbar";
import { currentUser } from "@/lib/auth";
import { meshName } from "@/lib/config";

import TrainingDock from "@/components/dashboard/TrainingDock";

import "./dashboard.css";
import "./radar.css";

export const dynamic = "force-dynamic";

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
  const user = await currentUser();
  // Guarding in the layout rather than middleware keeps session verification on
  // the Node runtime, where the signing key lives on disk.
  if (!user) redirect("/login");

  return (
    <MeshProvider>
      <div className="dash">
        <Sidebar
          meshName={meshName()}
          user={{ name: user.name, email: user.email, role: user.role }}
        />
        <div className="dash-main">
          <Topbar role={user.role} />
          <div className="dash-content">
            <SetupBanner />
            {children}
          </div>
          <TrainingDock />
        </div>
      </div>
    </MeshProvider>
  );
}
