"use server";

import { revalidatePath } from "next/cache";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";

import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

export async function cancelMeeting(data: FormData) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    throw new Error("User not found!");
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
    throw new Error("User not found!");
  }

  await nylas.events.destroy({
    eventId: data.get("eventId") as string,
    identifier: dbUser.grantId[0] as string,
    queryParams: {
      calendarId: dbUser.grantEmail[0] as string,
    },
  });

  revalidatePath("/dashboard/meetings");
}
