"use client";

import Image from "next/image";
import Link from "next/link";

import { Link2, Loader2, Trash } from "lucide-react";
import { toast } from "sonner";

import GoogleMeet from "@/assets/img/meet.svg";
import Teams from "@/assets/img/teams.svg";
import Zoom from "@/assets/img/zoom.svg";
import { env } from "@/env";
import useRefetch from "@/hooks/use-refetch";
import { cn } from "@/lib/utils";
import { api } from "@/trpc/react";

import { Button, buttonVariants } from "../ui/button";
import { Switch } from "../ui/switch";

interface EventCardProps {
  data: {
    id: string;
    title: string;
    duration: number;
    url: string;
    description: string;
    videoCallSoftware: string;
    createdAt: Date;
    updatedAt: Date;
    userId: string | null;
    active: boolean;
  };
}

const EventCard = ({ data }: EventCardProps) => {
  const refetch = useRefetch();
  const deleteEvent = api.event.deleteEventType.useMutation();

  const handleDelete = (id: string) => {
    deleteEvent.mutate(
      {
        eventId: id,
      },
      {
        onSuccess: () => {
          toast.success("Successfully Deleted Event!");
          refetch();
        },
        onError: () => {
          toast.error("Failed to Delete Event!");
        },
      }
    );
  };

  return (
    <div className="col-span-1 flex w-full flex-col items-start justify-start overflow-hidden rounded-lg border">
      <div className="flex w-full items-center justify-center gap-5 bg-sidebar p-2.5">
        <Image
          src={
            data.videoCallSoftware === "Google Meet"
              ? GoogleMeet
              : data.videoCallSoftware === "Zoom Meeting"
                ? Zoom
                : Teams
          }
          alt="meeting-service-icon"
          width={40}
          height={40}
          className="size-10 shrink-0"
        />
        <div className="flex w-full flex-col items-center justify-center">
          <span className="w-full overflow-hidden truncate text-left font-semibold">
            {data.title}
          </span>
          <span className="w-full text-left text-sm text-gray-500">
            Scheduled for {data.duration} Minutes
          </span>
        </div>
        <Switch defaultChecked={data.active} className="shrink-0" />
      </div>
      <div className="flex w-full flex-col items-center justify-center p-2.5">
        <span className="line-clamp-3 w-full text-left text-sm text-gray-500">
          {data.description}
        </span>
      </div>
      <div className="flex w-full items-center justify-end gap-2.5 border-t p-2.5">
        <Link
          href={`${env.NEXT_PUBLIC_APP_URL}/${data.url}`}
          className={cn(
            buttonVariants({
              variant: "outline",
              size: "icon",
            })
          )}
        >
          <Link2 />
        </Link>
        <Button
          type="button"
          variant="destructive"
          size="icon"
          disabled={deleteEvent.isPending}
          className="border-none"
          onClick={() => handleDelete(data.id)}
        >
          {deleteEvent.isPending ? (
            <Loader2 className="animate-spin" />
          ) : (
            <Trash />
          )}
        </Button>
      </div>
    </div>
  );
};

export default EventCard;
