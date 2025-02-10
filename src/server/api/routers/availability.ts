import { createTRPCRouter, privateProcedure } from "../trpc";

export const availabilityRouter = createTRPCRouter({
  createDefaultAvailability: privateProcedure.mutation(async ({ ctx }) => {
    return await ctx.db.availability.createMany({
      data: [
        {
          userId: ctx.user.id,
          day: "Monday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
        {
          userId: ctx.user.id,
          day: "Tuesday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
        {
          userId: ctx.user.id,
          day: "Wednesday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
        {
          userId: ctx.user.id,
          day: "Thursday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
        {
          userId: ctx.user.id,
          day: "Friday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
        {
          userId: ctx.user.id,
          day: "Saturday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
        {
          userId: ctx.user.id,
          day: "Sunday",
          fromTime: "08:00",
          tillTime: "18:00",
        },
      ],
    });
  }),
});
