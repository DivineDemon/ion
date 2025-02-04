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
  title: "Create T3 App",
  description: "Generated by create-t3-app",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <AuthProvider>
      <html lang="en" className={cn(dmSans.variable, eb_garamond.variable)}>
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
