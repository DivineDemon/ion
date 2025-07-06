"use client";

import { EllipsisVertical, ExternalLink, Link2, Pen, Trash } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { startTransition, useOptimistic, useState } from "react";
import { toast } from "sonner";

import { deleteEventType } from "@/app/(server-actions)/delete-event-type";
import { toggleEventType } from "@/app/(server-actions)/toggle-event-type";
import GoogleMeet from "@/assets/img/meet.svg";
import Teams from "@/assets/img/teams.svg";
import Zoom from "@/assets/img/zoom.svg";
import { env } from "@/env";
import { copyToClipboard } from "@/lib/utils";
import { api } from "@/trpc/react";

import { Button } from "../ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { Switch } from "../ui/switch";
import EditEventModal from "./edit-event-modal";

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
  const [edit, setEdit] = useState<boolean>(false);
  const [optimisticEventType, toggleOptimisticEventType] = useOptimistic(data, (state, newState: boolean) => {
    return { ...state, active: newState };
  });

  const { data: user } = api.user.findUser.useQuery();

  const handleDelete = async (id: string) => {
    const response = await deleteEventType(id);

    if (response.success) {
      toast.success("Successfully Deleted Event!");
    } else {
      toast.error("Failed to Delete Event!");
    }
  };

  const handleToggle = async () => {
    startTransition(() => {
      toggleOptimisticEventType(!optimisticEventType.active);
    });

    try {
      const response = await toggleEventType(data.id, !optimisticEventType.active);

      if (!response.success) {
        toast.error("Failed to Toggle Event!");
        startTransition(() => toggleOptimisticEventType(optimisticEventType.active));
      } else {
        toast.success("Successfully Toggled Event!");
      }
    } catch (error) {
      toast.error((error as Error).message);
      startTransition(() => toggleOptimisticEventType(optimisticEventType.active));
    }
  };

  return (
    <>
      <EditEventModal open={edit} setOpen={setEdit} id={optimisticEventType.id} />
      <div className="col-span-1 flex h-fit w-full flex-col items-start justify-start overflow-hidden rounded-lg border">
        <div className="flex w-full items-center justify-center gap-5 bg-sidebar p-2.5">
          <Image
            src={
              optimisticEventType.videoCallSoftware === "Google Meet"
                ? GoogleMeet
                : optimisticEventType.videoCallSoftware === "Zoom Meeting"
                  ? Zoom
                  : Teams
            }
            alt="meeting-service-icon"
            width={40}
            height={40}
            className="size-10 shrink-0"
          />
          <div className="flex w-full flex-col items-center justify-center">
            <span className="w-full overflow-hidden truncate text-left font-semibold">{optimisticEventType.title}</span>
            <span className="w-full text-left text-gray-500 text-sm">
              Scheduled for {optimisticEventType.duration} Minutes
            </span>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button type="button" variant="outline" className="size-10">
                <EllipsisVertical />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-44" align="end">
              <DropdownMenuLabel>Event Type</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuGroup>
                <DropdownMenuItem>
                  <Link
                    href={`${env.NEXT_PUBLIC_APP_URL}/${user?.userName}/${optimisticEventType.url}`}
                    target="_blank"
                    className="flex w-full items-center justify-start gap-4"
                  >
                    <ExternalLink className="size-4" />
                    <span>Preview</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() =>
                    copyToClipboard(`${env.NEXT_PUBLIC_APP_URL}/${user?.userName}/${optimisticEventType.url}`)
                  }
                >
                  <Link2 />
                  <span className="ml-2">Copy</span>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <div onClick={() => setEdit(true)} className="flex w-full items-center justify-start gap-4">
                    <Pen className="size-4" />
                    <span>Edit</span>
                  </div>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="group" onClick={() => handleDelete(optimisticEventType.id)}>
                  <Trash className="transition-colors group-hover:text-destructive" />
                  <span className="ml-2">Delete</span>
                </DropdownMenuItem>
              </DropdownMenuGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <div className="flex w-full flex-col items-center justify-center p-2.5">
          <span className="line-clamp-3 w-full text-left text-gray-500 text-sm">{optimisticEventType.description}</span>
        </div>
        <div className="flex w-full items-center justify-between gap-2.5 border-t p-2.5">
          <Switch className="shrink-0" onCheckedChange={handleToggle} defaultChecked={optimisticEventType.active} />
        </div>
      </div>
    </>
  );
};

export default EventCard;
