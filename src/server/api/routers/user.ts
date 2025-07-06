import { userSchema } from "@/lib/validators";
import { createTRPCRouter, privateProcedure, publicProcedure } from "@/server/api/trpc";

export const userRouter = createTRPCRouter({
  syncUser: publicProcedure.input(userSchema).mutation(async ({ ctx, input }) => {
    return ctx.db.user.upsert({
      where: {
        email: input.email,
      },
      update: {
        imageUrl: input.imageUrl,
        firstName: input.firstName,
        lastName: input.lastName,
      },
      create: {
        id: input.id,
        email: input.email,
        imageUrl: input.imageUrl,
        firstName: input.firstName,
        lastName: input.lastName,
        availability: {
          createMany: {
            data: [
              {
                day: "Monday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
              {
                day: "Tuesday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
              {
                day: "Wednesday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
              {
                day: "Thursday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
              {
                day: "Friday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
              {
                day: "Saturday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
              {
                day: "Sunday",
                fromTime: "08:00",
                tillTime: "18:00",
              },
            ],
          },
        },
      },
    });
  }),
  findUser: privateProcedure.query(async ({ ctx }) => {
    return await ctx.db.user.findUnique({
      where: {
        id: ctx.user.id,
      },
    });
  }),
  updateUser: privateProcedure.input(userSchema).mutation(async ({ ctx, input }) => {
    return await ctx.db.user.update({
      where: {
        id: ctx.user.id,
      },
      data: {
        ...input,
      },
    });
  }),
  fetchUserLinkedAccounts: privateProcedure.query(async ({ ctx }) => {
    return await ctx.db.user.findUnique({
      where: {
        id: ctx.user.id,
      },
      select: {
        grantId: true,
        grantEmail: true,
      },
    });
  }),
});
