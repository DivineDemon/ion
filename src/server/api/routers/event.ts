import { postAppointmentSchema } from "@/lib/validators";

import { createTRPCRouter, privateProcedure } from "../trpc";

export const eventRouter = createTRPCRouter({
  createEventType: privateProcedure
    .input(postAppointmentSchema)
    .mutation(async ({ ctx, input }) => {
      return await ctx.db.eventType.create({
        data: {
          ...input,
          userId: ctx.user.id,
        },
      });
    }),
});
