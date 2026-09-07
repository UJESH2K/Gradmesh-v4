/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [
      // Short, memorable URLs for the one-line join commands. A contributor
      // types what they see on the Join page, nothing else.
      { source: "/join.ps1", destination: "/api/join/ps1" },
      { source: "/join.sh", destination: "/api/join/sh" },
      { source: "/agent.zip", destination: "/api/agent/bundle" },
    ];
  },
};

export default nextConfig;
