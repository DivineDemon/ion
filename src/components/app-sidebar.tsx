"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { Atom, Bot } from "lucide-react";

import { SIDEBAR_ITEMS } from "@/lib/constants";
import { cn } from "@/lib/utils";

import ChatBar from "./chat-bar";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "./ui/sidebar";

const AppSidebar = () => {
  const path = usePathname();
  const [open, setOpen] = useState<boolean>(false);

  return (
    <>
      <ChatBar open={open} setOpen={setOpen} />
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
        <SidebarFooter
          onClick={() => setOpen(true)}
          className="flex w-full cursor-pointer flex-row items-center justify-center gap-2.5 border-t"
        >
          <div className="size-10 shrink-0 rounded-full bg-primary/20 p-2 text-yellow-600">
            <Bot className="size-full" />
          </div>
          <div className="flex flex-1 flex-col items-center justify-center">
            <span className="w-full text-left font-medium text-primary">
              Talk to IonBot.
            </span>
            <span className="w-full text-left text-xs text-muted-foreground">
              Your Personal Assistant.
            </span>
          </div>
        </SidebarFooter>
      </Sidebar>
    </>
  );
};

export default AppSidebar;
