"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import {
  type CalendarDate,
  getLocalTimeZone,
  parseDate,
  today,
} from "@internationalized/date";
import { DateValue } from "@react-types/calendar";

import Calendar from "../ui/calendar";

interface CustomCalendarProps {
  availability: {
    day: string;
    isActive: boolean;
  }[];
}

const CustomCalendar = ({ availability }: CustomCalendarProps) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [date, setDate] = useState<CalendarDate>(() => {
    const dateParam = searchParams.get("date");
    return dateParam ? parseDate(dateParam) : today(getLocalTimeZone());
  });

  const isDateUnavailable = (date: DateValue) => {
    const dayOfWeek = date.toDate(getLocalTimeZone()).getDay();

    return !availability[dayOfWeek - 1]?.isActive;
  };

  const handleDateChange = (date: DateValue) => {
    setDate(date as CalendarDate);
    const url = new URL(window.location.href);
    url.searchParams.set("date", date.toString());
    router.push(url.toString());
  };

  useEffect(() => {
    const dateParam = searchParams.get("date");

    if (dateParam) {
      setDate(parseDate(dateParam));
    }
  }, [searchParams]);

  return (
    <Calendar
      value={date}
      onChange={handleDateChange}
      minValue={today(getLocalTimeZone())}
      isDateUnavailable={isDateUnavailable}
    />
  );
};

export default CustomCalendar;
