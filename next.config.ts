import "./src/env.ts";

/** @type {import("next").NextConfig} */
const config = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "avatars.githubusercontent.com",
      },
      {
        protocol: "https",
        hostname: "img.clerk.com",
      },
      {
        protocol: "https",
        hostname: "lh3.googleusercontent.com",
      },
      {
        protocol: "https",
        hostname: "gravatar.com",
      },
      {
        protocol: "https",
        hostname: "ui.shadcn.com",
      },
    ],
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "3mb",
    },
  },
};

export default config;
