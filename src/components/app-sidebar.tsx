"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { Atom } from "lucide-react";

import { SIDEBAR_ITEMS } from "@/lib/constants";
import { cn } from "@/lib/utils";

import {
  Sidebar,
  SidebarContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "./ui/sidebar";

const AppSidebar = () => {
  const path = usePathname();

  return (
    <Sidebar>
      <SidebarContent className="flex w-full flex-col items-center justify-start">
        <SidebarMenu className="gap-0 space-y-0">
          <div className="flex w-full items-center justify-start border-b px-5 py-2.5">
            <Atom className="size-7 text-primary" />
          </div>
          <div className="flex w-full flex-col items-center justify-start gap-2.5 p-2.5">
            {SIDEBAR_ITEMS.map((item, idx) => (
              <SidebarMenuItem
                key={idx}
                className={cn("w-full rounded-md", {
                  "bg-primary/20 text-yellow-600": path === item.href,
                })}
              >
                <SidebarMenuButton asChild>
                  <Link href={item.href}>
                    <item.icon
                      className={cn({
                        "text-yellow-600": path === item.href,
                      })}
                    />
                    <span>{item.text}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </div>
        </SidebarMenu>
      </SidebarContent>
    </Sidebar>
  );
};

export default AppSidebar;
