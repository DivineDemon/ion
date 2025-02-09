import { CalendarCheck2, Home, Key, Settings, UsersRound } from "lucide-react";

export const SIDEBAR_ITEMS = [
  { href: "/dashboard", icon: Home, text: "Dashboard" },
  { href: "/dashboard/meetings", icon: UsersRound, text: "Meetings" },
  {
    href: "/dashboard/availability",
    icon: CalendarCheck2,
    text: "Availability",
  },
  {
    href: "/dashboard/settings",
    icon: Settings,
    text: "Settings",
  },
];
