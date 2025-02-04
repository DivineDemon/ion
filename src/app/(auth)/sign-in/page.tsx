"use client";

import Link from "next/link";
import { useState } from "react";

import { LoginLink } from "@kinde-oss/kinde-auth-nextjs";
import { Atom } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { env } from "@/env";
import { cn } from "@/lib/utils";

const Page = () => {
  const [email, setEmail] = useState<string>("");

  return (
    <div className="flex h-screen w-full flex-col items-center justify-center gap-5">
      <form className="flex w-1/5 flex-col items-center justify-center gap-5 p-5">
        <Atom className="size-12 text-primary" />
        <span className="w-full text-center text-2xl font-semibold">
          Sign in to <span className="text-primary">Ion</span>
        </span>
        <div className="flex w-full flex-col items-center justify-center gap-1.5">
          <Label
            htmlFor="email"
            className="w-full text-left text-xs font-medium"
          >
            Email
          </Label>
          <Input
            placeholder="johndoe@example.com"
            type="email"
            className="w-full"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <LoginLink
          className={cn("w-full", buttonVariants({ variant: "default" }))}
          authUrlParams={{
            connection_id: env.NEXT_PUBLIC_KINDE_EMAIL_CONNECTION_ID,
            login_hint: email,
          }}
        >
          Login
        </LoginLink>
        <fieldset className="w-full border-t border-gray-300 text-center">
          <legend className="px-5">or</legend>
        </fieldset>
        <div className="flex w-full flex-col items-center justify-center gap-2.5">
          <LoginLink
            className={cn("w-full", buttonVariants({ variant: "default" }))}
            authUrlParams={{
              connection_id: env.NEXT_PUBLIC_KINDE_GOOGLE_CONNECTION_ID,
            }}
          >
            Login with Google
          </LoginLink>
          <LoginLink
            className={cn("w-full", buttonVariants({ variant: "default" }))}
            authUrlParams={{
              connection_id: env.NEXT_PUBLIC_KINDE_GITHUB_CONNECTION_ID,
            }}
          >
            Login with Github
          </LoginLink>
        </div>
        <Link href="/sign-up" className="w-full text-center text-xs">
          Don&apos;t have an account ?&nbsp;
          <span className="cursor-pointer font-medium text-primary underline">
            Sign up
          </span>
        </Link>
      </form>
    </div>
  );
};

export default Page;
