import { type ReactNode } from "react";

import { Atom } from "lucide-react";

import MaxWidthWrapper from "@/components/max-width-wrapper";
import ModeToggle from "@/components/mode-toggle";
import { ThemeProvider } from "@/components/theme-provider";

const Layout = async ({ children }: { children: ReactNode }) => {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="flex h-screen w-full flex-col items-center justify-start">
        <nav className="flex h-[49px] w-full items-center justify-between border-b bg-sidebar pl-5 pr-2.5">
          <Atom className="text-primary" />
          <ModeToggle />
        </nav>
        <div className="h-[calc(100vh-49px)] w-full">
          <MaxWidthWrapper>{children}</MaxWidthWrapper>
        </div>
      </div>
    </ThemeProvider>
  );
};

export default Layout;
