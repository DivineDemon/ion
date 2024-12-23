import { currentUser } from "@clerk/nextjs/server";

import { db } from "@/db";

import { router } from "../__internals/router";
import { publicProcedure } from "../procedures";

export const userRouter = router({
  getDatabaseUser: publicProcedure.query(async ({ c }) => {
    const auth = await currentUser();

    if (!auth) {
      throw new Error("User not Authenticated!");
    }

    const user = await db.user.findUnique({
      where: {
        externalId: auth.id,
      },
    });

    if (!user) {
      throw new Error("User not Found!");
    }

    return c.json({
      user,
    });
  }),
});
