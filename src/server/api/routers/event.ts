import { z } from "zod";

import {
  postAppointmentSchema,
  updateAppointmentSchema,
} from "@/lib/validators";

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
  updateEventType: privateProcedure
    .input(updateAppointmentSchema)
    .mutation(async ({ ctx, input }) => {
      return ctx.db.eventType.update({
        where: {
          id: input.id,
          userId: ctx.user.id,
        },
        data: {
          videoCallSoftware: input.videoCallSoftware,
          url: input.url,
          description: input.description,
          duration: input.duration,
          title: input.title,
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
  getEventType: privateProcedure
    .input(
      z.object({
        eventId: z.string(),
      })
    )
    .query(async ({ ctx, input }) => {
      return await ctx.db.eventType.findUnique({
        where: {
          id: input.eventId,
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
  toggleEventType: privateProcedure
    .input(
      z.object({
        eventId: z.string(),
        active: z.boolean(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.eventType.update({
        where: {
          userId: ctx.user.id,
          id: input.eventId,
        },
        data: {
          active: input.active,
        },
      });
    }),
});
