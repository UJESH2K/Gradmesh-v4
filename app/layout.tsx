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
    // suppressHydrationWarning covers the <html> element only. Browser
    // extensions such as screen recorders and password managers stamp
    // attributes onto it before React hydrates, which React then reports as a
    // mismatch it cannot repair. The warning is about the extension, not this
    // app, and suppressing it here does not hide mismatches in any child.
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <div className="shell">{children}</div>
      </body>
    </html>
  );
}
