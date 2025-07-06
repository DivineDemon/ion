"use server";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { revalidatePath } from "next/cache";

import { db } from "@/server/db";

export async function toggleEventType(id: string, active: boolean) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  try {
    await db.eventType.update({
      where: {
        id,
      },
      data: {
        active,
      },
    });

    revalidatePath("/dashboard");
    return { success: true };
  } catch (_error) {
    return { success: false, error: "Failed to toggle event" };
  }
}
