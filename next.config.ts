import "./src/env.ts";

/** @type {import("next").NextConfig} */
const config = {
  experimental: {
    serverActions: {
      bodySizeLimit: "3mb",
    },
  },
};

export default config;
