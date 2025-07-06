"use client";

import {
  type CalendarDate,
  createCalendar,
  type DateDuration,
  endOfMonth,
  getLocalTimeZone,
  getWeeksInMonth,
  isSameMonth,
  isToday,
} from "@internationalized/date";
import { useButton } from "@react-aria/button";
import { useFocusRing } from "@react-aria/focus";
import { useDateFormatter } from "@react-aria/i18n";
import { mergeProps } from "@react-aria/utils";
import { VisuallyHidden } from "@react-aria/visually-hidden";
import type { CalendarProps, DateValue } from "@react-types/calendar";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useRef } from "react";
import { useCalendar, useCalendarCell, useCalendarGrid, useLocale } from "react-aria";
import { type CalendarState, useCalendarState } from "react-stately";

import { cn } from "@/lib/utils";

import { Button } from "./button";

const CalendarCell = ({
  state,
  date,
  currentMonth,
  isUnavailable,
}: {
  state: CalendarState;
  date: CalendarDate;
  currentMonth: CalendarDate;
  isUnavailable?: boolean;
}) => {
  const calendarCellRef = useRef<HTMLDivElement>(null);

  const {
    cellProps,
    buttonProps: calendarCellProps,
    formattedDate,
    isSelected,
    isDisabled,
  } = useCalendarCell({ date }, state, calendarCellRef);
  const finallyIsDisabled = isDisabled || isUnavailable;

  const { focusProps, isFocusVisible } = useFocusRing();

  return (
    <td
      {...cellProps}
      className={cn("relative z-0 p-0.5", {
        "z-10": isFocusVisible,
      })}
    >
      <div
        {...mergeProps(focusProps, calendarCellProps)}
        ref={calendarCellRef}
        hidden={!isSameMonth(currentMonth, date)}
        className="group size-12 rounded-md outline-none"
      >
        <div
          className={cn("relative flex size-full items-center justify-center rounded-sm font-semibold text-sm", {
            "bg-primary text-black": isSelected,
            "cursor-not-allowed text-muted-foreground": finallyIsDisabled,
            "hover:bg-primary/10": !isSelected && !finallyIsDisabled,
            "bg-secondary": !isSelected && !finallyIsDisabled,
          })}
        >
          {formattedDate}
          {isToday(date, getLocalTimeZone()) && (
            <div
              className={cn("absolute inset-x-0 bottom-1.5 mx-auto size-1.5 rounded-full bg-primary", {
                "bg-black": isSelected,
              })}
            />
          )}
        </div>
      </div>
    </td>
  );
};

const Calendar = (
  props: CalendarProps<DateValue> & {
    isDateUnavailable?: (date: DateValue) => boolean;
  },
) => {
  const { locale } = useLocale();

  const calendarButtonRef = useRef<HTMLButtonElement>(null);

  const state = useCalendarState({
    ...props,
    visibleDuration: {
      months: 1,
    },
    locale,
    createCalendar,
  });

  const { calendarProps, prevButtonProps, nextButtonProps } = useCalendar(props, state);

  const monthFormatter = useDateFormatter({
    month: "short",
    year: "numeric",
    timeZone: state.timeZone,
  });

  const [monthName, _, year] = monthFormatter
    .formatToParts(state.visibleRange.start.toDate(state.timeZone))
    .map((part) => part.value);

  const { buttonProps: prevProps } = useButton(prevButtonProps, calendarButtonRef);

  const { buttonProps: nextProps } = useButton(nextButtonProps, calendarButtonRef);

  const { focusProps } = useFocusRing();

  const offset: DateDuration = {};

  const weeksInMonth = getWeeksInMonth(state.visibleRange.start.add(offset), locale);

  const { gridProps, headerProps, weekDays } = useCalendarGrid(
    {
      startDate: state.visibleRange.start.add(offset),
      endDate: endOfMonth(state.visibleRange.start.add(offset)),
      weekdayStyle: "short",
    },
    state,
  );

  return (
    <div {...calendarProps} className="inline-block">
      <div className="flex items-center px-3 pb-4">
        <VisuallyHidden>
          <h2>{calendarProps["aria-label"]}</h2>
        </VisuallyHidden>
        <h2 className="flex-1 font-semibold">
          {monthName}&nbsp;
          <span className="font-medium text-muted-foreground text-sm">{year}</span>
        </h2>
        <div className="flex items-center gap-2">
          <Button
            ref={calendarButtonRef}
            variant="outline"
            size="icon"
            disabled={prevButtonProps.isDisabled}
            {...mergeProps(prevProps, focusProps)}
          >
            <ChevronLeft />
          </Button>
          <Button
            ref={calendarButtonRef}
            variant="outline"
            size="icon"
            disabled={nextButtonProps.isDisabled}
            {...mergeProps(nextProps, focusProps)}
          >
            <ChevronRight />
          </Button>
        </div>
      </div>
      <div className="flex gap-8">
        <table {...gridProps} cellPadding={0} className="flex-1">
          <thead {...headerProps} className="font-medium text-sm">
            <tr>
              {weekDays.map((day, index) => (
                <th key={index} className="pb-4">
                  {day}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...new Array(weeksInMonth).keys()].map((weekIndex) => (
              <tr key={weekIndex}>
                {state
                  .getDatesInWeek(weekIndex)
                  .map((date, i) =>
                    date ? (
                      <CalendarCell
                        key={i}
                        state={state}
                        date={date}
                        isUnavailable={props.isDateUnavailable?.(date)}
                        currentMonth={state.visibleRange.start.add(offset)}
                      />
                    ) : (
                      <td key={i} />
                    ),
                  )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Calendar;
