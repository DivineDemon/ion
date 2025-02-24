"use client";

import Image from "next/image";
import { useState } from "react";

import {
  LogoutLink,
  useKindeBrowserClient,
} from "@kinde-oss/kinde-auth-nextjs";
import { Bot, Loader2 } from "lucide-react";

import useAccount from "@/hooks/use-account";

import ChatBar from "./chat-bar";
import ModeToggle from "./mode-toggle";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Separator } from "./ui/separator";
import { SidebarTrigger } from "./ui/sidebar";

const Toolbar = () => {
  const { user } = useKindeBrowserClient();
  const [open, setOpen] = useState<boolean>(false);
  const { accounts, setAccount, account } = useAccount();

  return (
    <>
      <ChatBar open={open} setOpen={setOpen} />
      <nav className="flex h-[49px] w-full items-center justify-between border-b bg-sidebar px-2.5">
        <SidebarTrigger />
        <div className="flex items-center justify-center">
          {!accounts ? (
            <Loader2 className="animate-spin text-primary" />
          ) : (
            <Select value={account} onValueChange={setAccount}>
              <SelectTrigger className="w-[225px] bg-background hover:bg-muted">
                <SelectValue placeholder="Select an account" />
              </SelectTrigger>
              <SelectContent>
                {accounts?.grantEmail?.map((email) => (
                  <SelectItem key={email} value={email}>
                    {email}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <Button
            type="button"
            onClick={() => setOpen(true)}
            variant="outline"
            size="icon"
            className="mx-1.5"
            title="Talk to ION"
          >
            <Bot className="size-full" />
          </Button>
          <ModeToggle />
          <Separator
            orientation="vertical"
            className="mx-3.5 h-8 w-px border border-primary/20"
          />
          <LogoutLink>
            <Image
              src={user?.picture ?? "https://ui.shadcn.com/avatars/04.png"}
              alt="user-picture"
              width={40}
              height={40}
              className="size-8 rounded-full border"
            />
          </LogoutLink>
        </div>
      </nav>
    </>
  );
};

export default Toolbar;
