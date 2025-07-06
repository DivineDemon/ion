"use client";

import { RegisterLink } from "@kinde-oss/kinde-auth-nextjs";
import { Atom } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

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
        <span className="w-full text-center font-semibold text-2xl">
          Create an <span className="text-primary">account</span>
        </span>
        <div className="flex w-full flex-col items-center justify-center gap-1.5">
          <Label htmlFor="email" className="w-full text-left font-medium text-xs">
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
        <RegisterLink
          className={cn("w-full", buttonVariants({ variant: "default" }))}
          authUrlParams={{
            connection_id: env.NEXT_PUBLIC_KINDE_EMAIL_CONNECTION_ID,
            register_hint: email,
          }}
        >
          Register
        </RegisterLink>
        <fieldset className="w-full border-gray-300 border-t text-center">
          <legend className="px-5">or</legend>
        </fieldset>
        <div className="flex w-full flex-col items-center justify-center gap-2.5">
          <RegisterLink
            className={cn("w-full", buttonVariants({ variant: "default" }))}
            authUrlParams={{
              connection_id: env.NEXT_PUBLIC_KINDE_GOOGLE_CONNECTION_ID,
            }}
          >
            Register with Google
          </RegisterLink>
          <RegisterLink
            className={cn("w-full", buttonVariants({ variant: "default" }))}
            authUrlParams={{
              connection_id: env.NEXT_PUBLIC_KINDE_GITHUB_CONNECTION_ID,
            }}
          >
            Register with Github
          </RegisterLink>
        </div>
        <Link href="/sign-in" className="w-full text-center text-xs">
          Already have an account ?&nbsp;
          <span className="cursor-pointer font-medium text-primary underline">Sign in</span>
        </Link>
      </form>
    </div>
  );
};

export default Page;
