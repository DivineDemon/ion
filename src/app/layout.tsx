import { type Metadata } from "next";
import { DM_Sans, EB_Garamond } from "next/font/google";

import "@/assets/css/globals.css";
import { AuthProvider } from "@/components/auth-provider";
import { Toaster } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";
import { TRPCReactProvider } from "@/trpc/react";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
});

const eb_garamond = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-heading",
});

export const metadata: Metadata = {
  title: "ION",
  description: "Next Gen AI Powered Personal Assistant.",
  icons: [{ rel: "icon", url: "/favicon.svg" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <AuthProvider>
      <html
        lang="en"
        className={cn(dmSans.variable, eb_garamond.variable)}
        suppressHydrationWarning={true}
      >
        <body className="flex min-h-[calc(100vh-1px)] flex-col font-sans antialiased">
          <main className="relative flex flex-1 flex-col">
            <TRPCReactProvider>
              <Toaster richColors={true} />
              {children}
            </TRPCReactProvider>
          </main>
        </body>
      </html>
    </AuthProvider>
  );
}
