import { Context } from "hono";

import { type EnvType } from "@/env";

import { Procedure } from "./procedure";

const baseProcedure = new Procedure();

type MiddlewareFunction<T = object, R = void> = (params: {
  ctx: T;
  next: <B>(args: B) => Promise<B & T>;
  c: Context<{ Bindings: EnvType }>;
}) => Promise<R>;

/**
 * A helper to easily define middlewares and new procedures
 */

export const m2 = {
  middleware: <T = object, R = void>(
    fn: MiddlewareFunction<T, R>
  ): MiddlewareFunction<T, R> => {
    return fn;
  },
  procedure: baseProcedure,
};
