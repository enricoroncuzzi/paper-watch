import type { NextConfig } from "next";

// The page is private and never framed or embedded, so every response carries the same headers.
// No CSP here: the app renders no third-party script, and a CSP without nonce plumbing would
// break Next's own inline bootstrap script.
const securityHeaders = [
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "no-referrer" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
