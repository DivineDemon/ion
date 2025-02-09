import { Home, Key, Settings } from "lucide-react";

export const SIDEBAR_ITEMS = [
  { href: "/dashboard", icon: Home, text: "Dashboard" },
  { href: "/dashboard/api-key", icon: Key, text: "API Key" },
  {
    href: "/dashboard/account-settings",
    icon: Settings,
    text: "Account Settings",
  },
];
