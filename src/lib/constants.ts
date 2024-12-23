import {
  CalendarCheck,
  Gem,
  Home,
  LucideIcon,
  Settings,
  UsersRound,
} from "lucide-react";

interface SidebarItem {
  href: string;
  icon: LucideIcon;
  text: string;
}

export const SIDEBAR_ITEMS: SidebarItem[] = [
  { href: "/dashboard", icon: Home, text: "Dashboard" },
  { href: "/meetings", icon: UsersRound, text: "Meetings" },
  { href: "/availability", icon: CalendarCheck, text: "Availability" },
  { href: "/dashboard/upgrade", icon: Gem, text: "Upgrade" },
  {
    href: "/dashboard/account-settings",
    icon: Settings,
    text: "Account Settings",
  },
];
