"use server";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { redirect } from "next/navigation";

import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

export async function bookMeeting(data: FormData) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    redirect("/sign-in");
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

  const eventTypeData = await db.eventType.findUnique({
    where: {
      id: data.get("eventTypeId") as string,
    },
    select: {
      title: true,
      description: true,
    },
  });

  const startDate = new Date(`${data.get("eventDate")}T${data.get("fromTime")}:00`);
  const endDate = new Date(startDate.getTime() + Number(data.get("meetingLength")) * 60000);

  await nylas.events.create({
    identifier: dbUser.grantId[0] as string,
    requestBody: {
      title: eventTypeData?.title,
      description: eventTypeData?.description,
      when: {
        startTime: Math.floor(startDate.getTime() / 1000),
        endTime: Math.floor(endDate.getTime() / 1000),
      },
      conferencing: {
        autocreate: {},
        provider: data.get("provider") as "Google Meet" | "Zoom Meeting" | "Microsoft Teams" | "GoToMeeting" | "WebEx",
      },
      participants: [
        {
          name: data.get("name") as string,
          email: data.get("email") as string,
          status: "yes",
        },
      ],
    },
    queryParams: {
      calendarId: dbUser.grantEmail[0] as string,
      notifyParticipants: true,
    },
  });

  return redirect("/success");
}
