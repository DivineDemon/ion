import Image from "next/image";
import { notFound } from "next/navigation";

import dayjs from "dayjs";
import { CalendarX2, Clock } from "lucide-react";

import { bookMeeting } from "@/app/(server-actions)/book-meeting";
import GoogleMeet from "@/assets/img/meet.svg";
import Teams from "@/assets/img/teams.svg";
import Zoom from "@/assets/img/zoom.svg";
import CustomCalendar from "@/components/booking/custom-calendar";
import TimeTable from "@/components/booking/time-table";
import SubmitButton from "@/components/submit-button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { db } from "@/server/db";

interface PageProps {
  params: Promise<{
    userName: string;
    eventUrl: string;
  }>;
  searchParams: Promise<{ date?: string; time?: string }>;
}

const Page = async ({ params, searchParams }: PageProps) => {
  const { date, time } = await searchParams;
  const { eventUrl, userName } = await params;

  const response = await db.eventType.findFirst({
    where: {
      url: eventUrl,
      User: {
        userName: userName,
      },
      active: true,
    },
    select: {
      id: true,
      title: true,
      duration: true,
      description: true,
      videoCallSoftware: true,
      User: {
        select: {
          imageUrl: true,
          firstName: true,
          lastName: true,
          availability: {
            select: {
              day: true,
              isActive: true,
            },
          },
        },
      },
    },
  });

  if (!response) {
    return notFound();
  }

  const showForm = !!date && !!time;

  return (
    <div className="flex h-full w-full items-center justify-center">
      <Card className="w-full">
        <CardContent
          className={cn("grid w-full p-5", {
            "grid-cols-[1fr,auto,1fr,auto,1fr,auto]": !showForm,
            "grid-cols-[1fr,auto,1fr]": showForm,
          })}
        >
          <div className="col-span-1 flex w-full flex-col items-start justify-center gap-5 divide-y pr-5">
            <div className="flex w-full flex-col items-start justify-start gap-2.5">
              <Image
                src={response.User?.imageUrl as string}
                alt="user-dp"
                width={40}
                height={40}
                className="size-10 rounded-full"
              />
              <p className="w-full text-left text-sm font-medium text-muted-foreground">
                {response.User?.firstName}&nbsp;{response.User?.lastName}
              </p>
            </div>
            <div className="flex w-full flex-col items-start justify-start gap-2.5 pt-5">
              <h1 className="w-full text-left text-xl font-semibold">
                {response.title}
              </h1>
              <p className="w-full max-w-prose text-pretty text-left text-sm font-medium text-muted-foreground">
                {response.description}
              </p>
            </div>
            <div className="flex w-full flex-col items-start justify-start gap-2.5 pt-5">
              <div className="flex w-full items-center justify-center gap-3">
                <CalendarX2 className="size-4 text-primary" />
                <span className="flex-1 text-left text-sm font-medium text-muted-foreground">
                  {date
                    ? dayjs(date).format("DD MMMM, YYYY")
                    : dayjs(new Date()).format("DD MMMM, YYYY")}
                </span>
              </div>
              <div className="flex w-full items-center justify-center gap-3">
                <Clock className="size-4 text-primary" />
                <span className="flex-1 text-left text-sm font-medium text-muted-foreground">
                  {response.duration}&nbsp;Minutes
                </span>
              </div>
              <div className="flex w-full items-center justify-center gap-2">
                <Image
                  width={16}
                  height={16}
                  className="size-5"
                  alt="video-call-software"
                  src={
                    response.videoCallSoftware === "Google Meet"
                      ? GoogleMeet
                      : response.videoCallSoftware === "Zoom Meeting"
                        ? Zoom
                        : Teams
                  }
                />
                <span className="flex-1 text-left text-sm font-medium text-muted-foreground">
                  {response.videoCallSoftware}
                </span>
              </div>
            </div>
          </div>
          <Separator orientation="vertical" />
          {showForm ? (
            <form
              action={bookMeeting}
              className="col-span-1 flex w-full flex-col items-start justify-start gap-5 pl-5"
            >
              <Input type="hidden" name="fromTime" value={time} />
              <Input type="hidden" name="eventDate" value={date} />
              <Input type="hidden" name="eventTypeId" value={response.id} />
              <Input
                type="hidden"
                name="meetingLength"
                value={response.duration}
              />
              <Input
                type="hidden"
                name="provider"
                value={response.videoCallSoftware}
              />
              <div className="flex w-full flex-col items-center justify-center gap-2">
                <Label
                  htmlFor="name"
                  className="w-full text-left text-sm font-medium"
                >
                  Name
                </Label>
                <Input
                  className="w-full"
                  id="name"
                  type="text"
                  name="name"
                  placeholder="Your Name"
                />
              </div>
              <div className="flex w-full flex-col items-center justify-center gap-2">
                <Label
                  htmlFor="email"
                  className="w-full text-left text-sm font-medium"
                >
                  Email
                </Label>
                <Input
                  className="w-full"
                  id="email"
                  type="email"
                  name="email"
                  placeholder="johndoe@email.com"
                />
              </div>
              <SubmitButton text="Book Meeting" className="mt-auto w-full" />
            </form>
          ) : (
            <>
              <div className="col-span-1 flex w-full items-center justify-center px-5">
                <CustomCalendar availability={response.User?.availability!} />
              </div>
              <Separator orientation="vertical" />
              <div className="col-span-1 flex w-full items-center justify-center">
                <TimeTable
                  selectedDate={date ? new Date(date) : new Date()}
                  duration={response.duration}
                  time={time ?? ""}
                />
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default Page;
