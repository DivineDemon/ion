"use client";

import { getLocalTimeZone, today } from "@internationalized/date";
import { DateValue } from "@react-types/calendar";

import Calendar from "../ui/calendar";

interface CustomCalendarProps {
  availability: {
    day: string;
    isActive: boolean;
  }[];
}

const CustomCalendar = ({ availability }: CustomCalendarProps) => {
  const isDateUnavailable = (date: DateValue) => {
    const dayOfWeek = date.toDate(getLocalTimeZone()).getDay();

    return !availability[dayOfWeek - 1]?.isActive;
  };

  return (
    <Calendar
      minValue={today(getLocalTimeZone())}
      isDateUnavailable={isDateUnavailable}
    />
  );
};

export default CustomCalendar;
