"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";

import { client } from "@/lib/client";
import { cn } from "@/lib/utils";

import { Button, buttonVariants } from "./ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";

const NylasAuthModal = () => {
  // Modal Behaviour State
  const [open, setOpen] = useState<boolean>(false);

  // API to Fetch Nylas Integration Link
  const { data: link, isLoading } = useQuery({
    queryFn: async () => {
      const response = await client.nylas.getRedirectUrl.$get({});
      const { redirectUrl } = await response.json();

      return redirectUrl;
    },
    queryKey: ["get-nylas-url"],
  });

  // API to Fetch Currently Logged in User from the DB
  const { data } = useQuery({
    queryFn: async () => {
      const response = await client.user.getDatabaseUser.$get({});
      return await response.json();
    },
    queryKey: ["get-database-user"],
  });

  useEffect(() => {
    if (data) {
      if (data.user.grantId) {
        setOpen(false);
      } else {
        setOpen(true);
      }
    }
  }, [data]);

  return (
    <Dialog open={open}>
      <DialogContent className="overflow-hidden p-5">
        <DialogHeader>
          <DialogTitle>
            <span className="w-full text-left text-3xl">
              Almost <span className="text-primary">Done</span> . . .
            </span>
          </DialogTitle>
        </DialogHeader>
        <img
          src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExdng5YjNleGszZm83enBxbjdvYzJ6M2FmbnNibmRnc2ZsMGVzNm95NyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/AoiPYyV8TWL3T35HH1/giphy.webp"
          alt="cyberpunk-gif"
          className="w-full"
        />
        <div className="grid w-full grid-cols-2 gap-5">
          <Link
            href={`${link}`}
            className={cn(
              buttonVariants({
                variant: "default",
                size: "lg",
                className: "w-full",
              })
            )}
          >
            {isLoading ? (
              <Loader2 className="animate-spin" />
            ) : (
              "Connect your Calendar"
            )}
          </Link>
          <Button
            type="button"
            variant="secondary"
            size="lg"
            onClick={() => setOpen(false)}
          >
            Already Connected
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default NylasAuthModal;
