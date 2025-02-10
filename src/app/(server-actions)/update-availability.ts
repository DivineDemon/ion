"use server";

import { revalidatePath } from "next/cache";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";

import { db } from "@/server/db";

export async function updateAvailability(data: FormData) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    throw new Error("User not authenticated!");
  }

  const rawData = Object.fromEntries(data.entries());
  const availabilityData = Object.keys(rawData)
    .filter((key) => key.startsWith("id-"))
    .map((key) => {
      const id = key.replace("id-", "");

      return {
        id,
        isActive: rawData[`isActive-${id}`] === "on",
        fromTime: rawData[`fromTime-${id}`] as string,
        tillTime: rawData[`tillTime-${id}`] as string,
      };
    });

  try {
    await db.$transaction(
      availabilityData.map((availability) =>
        db.availability.update({
          where: {
            id: availability.id,
          },
          data: {
            isActive: availability.isActive,
            fromTime: availability.fromTime,
            tillTime: availability.tillTime,
          },
        })
      )
    );

    revalidatePath("/dashboard/availability");
  } catch (error: Error | unknown) {
    throw new Error((error as Error).message);
  }
}
