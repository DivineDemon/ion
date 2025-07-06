"use server";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { revalidatePath } from "next/cache";

import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

export async function getAllEvents(grantEmail: string) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const dbUser = await db.user.findUnique({
    where: {
      id: user.id,
    },
    select: {
      grantId: true,
      grantEmail: true,
    },
  });

  if (!dbUser) {
    return { success: false, error: "User not found" };
  }

  const index = dbUser.grantEmail.findIndex((email) => email === grantEmail);
  const grantId = index !== -1 ? dbUser.grantId[index] : null;

  const data = await nylas.events.list({
    identifier: grantId as string,
    queryParams: {
      calendarId: grantEmail as string,
    },
  });

  return data.data;
}

export async function deleteEvent(grantEmail: string, eventId: string) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return { success: false, error: "Unauthorized" };
  }

  const dbUser = await db.user.findUnique({
    where: {
      id: user.id,
    },
    select: {
      grantId: true,
      grantEmail: true,
    },
  });

  if (!dbUser) {
    return { success: false, error: "User not found" };
  }

  const index = dbUser.grantEmail.findIndex((email) => email === grantEmail);
  const grantId = index !== -1 ? dbUser.grantId[index] : null;

  await nylas.events.destroy({
    eventId: eventId as string,
    identifier: grantId as string,
    queryParams: {
      calendarId: grantEmail as string,
    },
  });

  revalidatePath("/dashboard/meetings");
}
