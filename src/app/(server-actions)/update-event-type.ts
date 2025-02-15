"use server";

import { revalidatePath } from "next/cache";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { z } from "zod";

import { appointmentTypeSchema } from "@/lib/validators";
import { db } from "@/server/db";

export async function updateEventType(
  id: string,
  values: z.infer<typeof appointmentTypeSchema>
) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  try {
    const validatedData = appointmentTypeSchema.parse(values);

    await db.eventType.update({
      where: {
        id,
      },
      data: {
        userId: user.id,
        ...validatedData,
        duration: Number(validatedData.duration),
      },
    });

    revalidatePath("/dashboard");
    return { success: true };
  } catch (error) {
    console.error(error);
    return { success: false, error: "Failed to update event" };
  }
}
