import Link from "next/link";
import { redirect } from "next/navigation";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import dayjs from "dayjs";
import { Video } from "lucide-react";

import { cancelMeeting } from "@/app/(server-actions)/cancel-meeting";
import NotFound from "@/components/not-found";
import SubmitButton from "@/components/submit-button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

const Page = async () => {
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

  const data = await nylas.events.list({
    identifier: dbUser.grantId[0] as string,
    queryParams: {
      calendarId: dbUser.grantEmail[0] as string,
    },
  });

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-5 p-5">
      {data.data.length < 1 ? (
        <NotFound
          title="No Meetings Found!"
          description="scheduling a meeting"
        />
      ) : (
        <Card className="flex h-full w-1/2 flex-col">
          <CardHeader className="w-full rounded-t-lg border-b bg-sidebar">
            <CardTitle className="text-2xl text-primary">Meetings</CardTitle>
            <CardDescription>
              Manage your meetings and schedule here.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex w-full flex-col items-start justify-start gap-5 divide-y px-5 pb-5">
            {data.data.map((event) => (
              <form
                action={cancelMeeting}
                className="grid w-full grid-cols-3 items-center justify-center pt-5"
                key={event.id}
              >
                <input type="hidden" name="eventId" value={event.id} />
                <div className="col-span-1 flex h-full w-full flex-col items-start justify-start">
                  <span className="w-full text-left text-sm text-muted-foreground">
                    {/* @ts-ignore */}
                    {dayjs.unix(event.when.startTime).format("ddd, DD MMM")}
                  </span>
                  <span className="w-full text-left text-xs text-muted-foreground">
                    {/* @ts-ignore */}
                    {dayjs.unix(event.when.startTime).format("HH:mm a")}
                    &nbsp;-&nbsp;
                    {/* @ts-ignore */}
                    {dayjs.unix(event.when.endTime).format("HH:mm a")}
                  </span>
                  <div className="mt-auto flex w-full items-center justify-start gap-2.5 pt-2">
                    <Video className="size-4 text-primary" />
                    <Link
                      target="_blank"
                      // @ts-ignore
                      href={event.conferencing.details.url}
                      className="flex-1 text-left text-sm text-primary hover:underline hover:underline-offset-4"
                    >
                      Join Meeting
                    </Link>
                  </div>
                </div>
                <div className="col-span-1 flex h-full w-full flex-col items-start justify-start gap-2">
                  <span className="w-full text-left text-sm font-medium">
                    {event.title}
                  </span>
                  <div className="flex w-full flex-wrap items-center justify-start gap-2.5">
                    <span className="rounded-full bg-primary/20 px-3 py-1 text-xs text-yellow-600">
                      You
                    </span>
                    {event.participants.length > 0 &&
                      event.participants.map((participant, idx) => (
                        <span
                          key={idx}
                          className="rounded-full bg-primary/20 px-3 py-1 text-xs text-yellow-600"
                        >
                          {participant?.email}
                        </span>
                      ))}
                  </div>
                </div>
                <div className="col-span-1 flex h-full w-full flex-col items-end justify-center">
                  <SubmitButton variant="destructive" text="Cancel Meeting" />
                </div>
              </form>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Page;
