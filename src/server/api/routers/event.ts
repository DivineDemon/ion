import { z } from "zod";

import { postAppointmentSchema } from "@/lib/validators";

import { createTRPCRouter, privateProcedure } from "../trpc";

export const eventRouter = createTRPCRouter({
  createEventType: privateProcedure
    .input(postAppointmentSchema)
    .mutation(async ({ ctx, input }) => {
      return ctx.db.eventType.create({
        data: {
          ...input,
          userId: ctx.user.id,
        },
      });
    }),
  getEventTypes: privateProcedure.query(async ({ ctx }) => {
    return await ctx.db.eventType.findMany({
      where: {
        userId: ctx.user.id,
      },
    });
  }),
  deleteEventType: privateProcedure
    .input(
      z.object({
        eventId: z.string(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.eventType.delete({
        where: {
          userId: ctx.user.id,
          id: input.eventId,
        },
      });
    }),
});
