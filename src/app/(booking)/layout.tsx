"use client";

import { Atom, Bot } from "lucide-react";
import { type ReactNode, useState } from "react";

import ChatBar from "@/components/chat-bar";
import MaxWidthWrapper from "@/components/max-width-wrapper";
import ModeToggle from "@/components/mode-toggle";
import { ThemeProvider } from "@/components/theme-provider";
import { Button } from "@/components/ui/button";

const Layout = ({ children }: { children: ReactNode }) => {
  const [open, setOpen] = useState<boolean>(false);

  return (
    <>
      <ChatBar open={open} setOpen={setOpen} />
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
        <div className="flex h-screen w-full flex-col items-center justify-start">
          <nav className="flex h-[49px] w-full items-center justify-between border-b bg-sidebar pr-2.5 pl-5">
            <Atom className="text-primary" />
            <div className="flex items-center justify-center gap-1.5">
              <Button onClick={() => setOpen(true)} type="button" variant="outline" size="icon">
                <Bot className="size-full" />
              </Button>
              <ModeToggle />
            </div>
          </nav>
          <div className="h-[calc(100vh-49px)] w-full">
            <MaxWidthWrapper>{children}</MaxWidthWrapper>
          </div>
        </div>
      </ThemeProvider>
    </>
  );
};

export default Layout;
