import { db } from "@/db";
import { userValidator } from "@/lib/validators";

import { router } from "../__internals/router";
import { privateProcedure } from "../procedures";

export const userRouter = router({
  getDatabaseUser: privateProcedure.query(async ({ c, ctx }) => {
    const user = await db.user.findUnique({
      where: {
        externalId: ctx.user.externalId!,
      },
    });

    if (!user) {
      throw new Error("User not Found!");
    }

    return c.json({
      user,
    });
  }),
  updateUser: privateProcedure
    .input(userValidator)
    .mutation(async ({ c, ctx, input }) => {
      await db.user.update({
        where: {
          externalId: ctx.user.externalId!,
        },
        data: {
          ...input,
        },
      });

      return c.json({
        success: true,
      });
    }),
});
