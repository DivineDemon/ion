"use server";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { revalidatePath } from "next/cache";

import { db } from "@/server/db";

export async function deleteEventType(eventId: string) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  try {
    await db.eventType.delete({
      where: {
        id: eventId,
      },
    });

    revalidatePath("/dashboard");
    return { success: true };
  } catch (_error) {
    return { success: false, error: "Failed to delete event type" };
  }
}
