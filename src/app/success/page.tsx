import Link from "next/link";

import { Check } from "lucide-react";

import { ThemeProvider } from "@/components/theme-provider";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const Page = () => {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="flex h-screen w-full items-center justify-center">
        <Card className="mx-auto w-full max-w-md">
          <CardContent className="flex w-full flex-col items-center p-5">
            <div className="flex size-16 items-center justify-center rounded-full bg-primary/10">
              <Check className="size-8 text-primary" />
            </div>
            <span className="mt-5 w-full text-center text-2xl font-semibold">
              This event is scheduled.
            </span>
            <span className="mt-2 w-full text-center text-sm text-muted-foreground">
              We emailed you a calendar invitation with all the details and the
              video call link!
            </span>
          </CardContent>
          <CardFooter className="w-full">
            <Link
              href="/dashboard"
              className={cn(
                "w-full",
                buttonVariants({
                  variant: "default",
                })
              )}
            >
              Close this Page
            </Link>
          </CardFooter>
        </Card>
      </div>
    </ThemeProvider>
  );
};

export default Page;
