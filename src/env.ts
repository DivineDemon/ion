/* eslint-disable n/no-process-env */
import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  server: {
    DATABASE_URL: z.string().url(),
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
    VERCEL_URL: z.string(),
    CLERK_SECRET_KEY: z.string().regex(/^sk_test_[A-Za-z0-9-\.]+$/, {
      message: "Invalid CLERK_SECRET_KEY format",
    }),
    NYLAS_API_KEY: z
      .string()
      .regex(/^nyk_v0_[A-Za-z0-9]{64}$/, "Invalid Nylas secret key format"),
    NYLAS_API_URL: z.string().url(),
    NYLAS_CLIENT_ID: z
      .string()
      .regex(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
        "Invalid UUID format"
      ),
  },
  client: {
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: z
      .string()
      .regex(/^pk_test_[A-Za-z0-9]+$/, {
        message: "Invalid NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY format",
      }),
    NEXT_PUBLIC_APP_URL: z.string(),
  },
  runtimeEnv: {
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY:
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY,
    CLERK_SECRET_KEY: process.env.CLERK_SECRET_KEY,
    VERCEL_URL: process.env.VERCEL_URL,
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    NYLAS_API_KEY: process.env.NYLAS_API_KEY,
    NYLAS_API_URL: process.env.NYLAS_API_URL,
    NYLAS_CLIENT_ID: process.env.NYLAS_CLIENT_ID,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  emptyStringAsUndefined: true,
});

export type EnvType = typeof env;
