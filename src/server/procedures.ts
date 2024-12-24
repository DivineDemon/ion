import { currentUser } from "@clerk/nextjs/server";
import { HTTPException } from "hono/http-exception";

import { db } from "@/db";

import { m2 } from "./__internals/m2";

const authMiddleware = m2.middleware(async ({ next }) => {
  const auth = await currentUser();

  if (!auth) {
    throw new HTTPException(401, { message: "Unauthorized" });
  }

  const user = await db.user.findUnique({
    where: { externalId: auth.id },
  });

  if (!user) {
    throw new HTTPException(401, { message: "Unauthorized" });
  }

  return next({ user });
});

export const baseProcedure = m2.procedure;
export const publicProcedure = baseProcedure;
export const privateProcedure = publicProcedure.use(authMiddleware);
