import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "GradMesh — every GPU on your network, one training cluster",
    template: "%s · GradMesh",
  },
  description:
    "GradMesh turns the idle GPUs already sitting on your network into a single coordinated training cluster. Share a link, contribute a GPU, train together.",
  applicationName: "GradMesh",
  openGraph: {
    title: "GradMesh",
    description: "Every GPU on your network, one training cluster.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#07080b",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">{children}</div>
      </body>
    </html>
  );
}
