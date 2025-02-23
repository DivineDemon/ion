import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  server: {
    DATABASE_URL: z.string().url(),
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
    KINDE_SITE_URL: z.string().url(),
    KINDE_ISSUER_URL: z.string().url(),
    KINDE_CLIENT_ID: z.string().min(1),
    KINDE_CLIENT_SECRET: z.string().min(1),
    KINDE_POST_LOGIN_REDIRECT_URL: z.string().url(),
    KINDE_POST_LOGOUT_REDIRECT_URL: z.string().url(),
    NYLAS_API_SECRET_KEY: z.string(),
    NYLAS_API_URI: z.string().url(),
    NYLAS_CLIENT_ID: z.string(),
    OPENAI_API_KEY: z.string(),
  },
  client: {
    NEXT_PUBLIC_KINDE_EMAIL_CONNECTION_ID: z.string().min(1),
    NEXT_PUBLIC_KINDE_GITHUB_CONNECTION_ID: z.string().min(1),
    NEXT_PUBLIC_KINDE_GOOGLE_CONNECTION_ID: z.string().min(1),
    NEXT_PUBLIC_APP_URL: z.string().url(),
    NEXT_PUBLIC_IMGBB_KEY: z.string(),
  },
  runtimeEnv: {
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    KINDE_SITE_URL: process.env.KINDE_SITE_URL,
    KINDE_ISSUER_URL: process.env.KINDE_ISSUER_URL,
    KINDE_CLIENT_ID: process.env.KINDE_CLIENT_ID,
    KINDE_CLIENT_SECRET: process.env.KINDE_CLIENT_SECRET,
    KINDE_POST_LOGIN_REDIRECT_URL: process.env.KINDE_POST_LOGIN_REDIRECT_URL,
    KINDE_POST_LOGOUT_REDIRECT_URL: process.env.KINDE_POST_LOGOUT_REDIRECT_URL,
    NEXT_PUBLIC_KINDE_EMAIL_CONNECTION_ID:
      process.env.NEXT_PUBLIC_KINDE_EMAIL_CONNECTION_ID,
    NEXT_PUBLIC_KINDE_GITHUB_CONNECTION_ID:
      process.env.NEXT_PUBLIC_KINDE_GITHUB_CONNECTION_ID,
    NEXT_PUBLIC_KINDE_GOOGLE_CONNECTION_ID:
      process.env.NEXT_PUBLIC_KINDE_GOOGLE_CONNECTION_ID,
    NYLAS_API_SECRET_KEY: process.env.NYLAS_API_SECRET_KEY,
    NYLAS_API_URI: process.env.NYLAS_API_URI,
    NYLAS_CLIENT_ID: process.env.NYLAS_CLIENT_ID,
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
    NEXT_PUBLIC_IMGBB_KEY: process.env.NEXT_PUBLIC_IMGBB_KEY,
    OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  emptyStringAsUndefined: true,
});
