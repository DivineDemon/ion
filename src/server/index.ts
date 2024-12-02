/* eslint-disable @typescript-eslint/no-unused-vars */
import { Hono } from "hono";
import { cors } from "hono/cors";
import { handle } from "hono/vercel";

import { authRouter } from "./routers/auth-router";

/**
 * Initializing our Hono app
 *
 * Defined our base path as /api, any route therefore, starting with /api will correspond to our backend
 * using cors() middleware provided by hono.js
 */
const app = new Hono().basePath("/api").use(cors());

/**
 * This is the primary router for your server.
 *
 * All routers added in /server/routers should be manually added here.
 */
const appRouter = app.route("/auth", authRouter);

/**
 * The handler Next.js uses to answer API requests
 *
 * Since we employ Hono.js for our type-safe backend implementation,
 * we need to tell Next.js to use our Hono app as the handler.
 */
export const httpHandler = handle(app);

export default app;
export type AppType = typeof appRouter;
