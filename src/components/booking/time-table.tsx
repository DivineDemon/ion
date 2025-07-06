import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { Prisma } from "@prisma/client";
import dayjs from "dayjs";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { GetFreeBusyResponse, NylasResponse } from "nylas";

import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

import { Button } from "../ui/button";

interface TimeTableProps {
  selectedDate: Date;
  duration: number;
  time: string;
}

const calculateAvailability = (
  date: string,
  duration: number,
  availability: {
    fromTime: string | undefined;
    tillTime: string | undefined;
  },
  nylasData: NylasResponse<GetFreeBusyResponse[]>,
) => {
  const now = dayjs();
  const availableFrom = dayjs(`${date} ${availability.fromTime}`);
  const availableTill = dayjs(`${date} ${availability.tillTime}`);

  if (!availableFrom.isValid() || !availableTill.isValid()) {
    return [];
  }

  const busySlots =
    // @ts-ignore
    nylasData.data[0]?.timeSlots?.map((slot) => ({
      start: dayjs.unix(slot.startTime),
      end: dayjs.unix(slot.endTime),
    })) || [];

  const allSlots: dayjs.Dayjs[] = [];
  let currentSlot = availableFrom;

  while (dayjs(currentSlot.add(duration, "minute")).isBefore(availableTill)) {
    allSlots.push(currentSlot);
    currentSlot = currentSlot.add(duration, "minute");
  }

  const freeSlots = allSlots.filter((slot) => {
    const slotEnd = slot.add(duration, "minute");
    const isAfterNow = slot.isAfter(now);

    // @ts-ignore
    const isSlotFree = !busySlots.some((busy) => {
      return (
        (slot.isAfter(busy.start) && slot.isBefore(busy.end)) ||
        (slotEnd.isAfter(busy.start) && slotEnd.isBefore(busy.end)) ||
        (slot.isBefore(busy.start) && slotEnd.isAfter(busy.end))
      );
    });

    return isAfterNow && isSlotFree;
  });

  return freeSlots.map((slot) => slot.format("HH:mm"));
};

const TimeTable = async ({ selectedDate, time, duration }: TimeTableProps) => {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  const response = await db.availability.findFirst({
    where: {
      day: dayjs(selectedDate).format("dddd") as Prisma.EnumDayFilter,
      userId: user.id,
    },
    select: {
      id: true,
      fromTime: true,
      tillTime: true,
      User: {
        select: {
          grantId: true,
          grantEmail: true,
        },
      },
    },
  });

  if (!response) {
    return notFound();
  }

  const startOfDay = new Date(selectedDate);
  startOfDay.setHours(0, 0, 0, 0);

  const endOfDay = new Date(selectedDate);
  endOfDay.setHours(23, 59, 59, 999);

  const nylasCalendarData = await nylas.calendars.getFreeBusy({
    identifier: response.User?.grantId[0] as string,
    requestBody: {
      startTime: Math.floor(startOfDay.getTime() / 1000),
      endTime: Math.floor(endOfDay.getTime() / 1000),
      emails: [response.User?.grantEmail[0] as string],
    },
  });

  const availableSlots = calculateAvailability(
    dayjs(selectedDate).format("YYYY-MM-DD"),
    duration,
    {
      fromTime: response.fromTime,
      tillTime: response.tillTime,
    },
    nylasCalendarData,
  );

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-5 pl-5">
      <p className="w-full text-left font-semibold text-[16px] leading-[16px]">
        {dayjs(selectedDate).format("ddd")}&nbsp;
        <span className="text-[14px] text-muted-foreground leading-[14px]">
          {dayjs(selectedDate).format("MMM. DD")}
        </span>
      </p>
      <div className="flex h-full max-h-[374px] w-full flex-col items-start justify-start gap-2 overflow-y-auto">
        {availableSlots.length > 0
          ? availableSlots.map((slot, idx) => (
              <Link
                href={`?date=${dayjs(selectedDate).format("YYYY-MM-DD")}&time=${slot}`}
                key={idx}
                className="w-full"
              >
                <Button type="button" className="w-full" variant={slot === time ? "default" : "outline"}>
                  {slot}
                </Button>
              </Link>
            ))
          : "No Available Time Slots"}
      </div>
    </div>
  );
};

export default TimeTable;
