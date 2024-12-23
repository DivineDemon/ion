import {
  CalendarCheck,
  Gem,
  Home,
  Key,
  LucideIcon,
  Settings,
  UsersRound,
} from "lucide-react";

interface SidebarItem {
  href: string;
  icon: LucideIcon;
  text: string;
}

interface SidebarCategory {
  category: string;
  items: SidebarItem[];
}

export const SIDEBAR_ITEMS: SidebarCategory[] = [
  {
    category: "Overview",
    items: [
      { href: "/dashboard", icon: Home, text: "Dashboard" },
      { href: "/meetings", icon: UsersRound, text: "Meetings" },
      { href: "/availability", icon: CalendarCheck, text: "Availability" },
    ],
  },
  {
    category: "Account",
    items: [{ href: "/dashboard/upgrade", icon: Gem, text: "Upgrade" }],
  },
  {
    category: "Settings",
    items: [
      { href: "/dashboard/api-key", icon: Key, text: "API Key" },
      {
        href: "/dashboard/account-settings",
        icon: Settings,
        text: "Account Settings",
      },
    ],
  },
];
