import Image from "next/image";
import { type ReactNode } from "react";

import {
  LogoutLink,
  getKindeServerSession,
} from "@kinde-oss/kinde-auth-nextjs/server";

import AppSidebar from "@/components/app-sidebar";
import ModeToggle from "@/components/mode-toggle";
import { ThemeProvider } from "@/components/theme-provider";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";

const Layout = async ({ children }: { children: ReactNode }) => {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <SidebarProvider>
        <div className="flex h-screen w-full items-center justify-center overflow-hidden">
          <AppSidebar />
          <div className="flex h-full w-full flex-1 flex-col items-center justify-start">
            <nav className="flex h-[49px] w-full items-center justify-between border-b bg-sidebar px-2.5">
              <SidebarTrigger />
              <div className="flex items-center justify-center gap-5">
                <ModeToggle />
                <LogoutLink>
                  <Image
                    src={user?.picture!}
                    alt="user-picture"
                    width={40}
                    height={40}
                    className="size-8 rounded-full border"
                  />
                </LogoutLink>
              </div>
            </nav>
            <div className="h-[calc(100vh-49px)] w-full">{children}</div>
          </div>
        </div>
      </SidebarProvider>
    </ThemeProvider>
  );
};

export default Layout;
