"use client";

import { type Dispatch, type SetStateAction, useEffect } from "react";

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2 } from "lucide-react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";

import { updateEventType } from "@/app/(server-actions)/update-event-type";
import { env } from "@/env";
import { appointmentTypeSchema } from "@/lib/validators";
import { api } from "@/trpc/react";

import { Button } from "../ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Textarea } from "../ui/textarea";

interface EditEventModalProps {
  id: string;
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
}

const EditEventModal = ({ id, open, setOpen }: EditEventModalProps) => {
  const form = useForm<z.infer<typeof appointmentTypeSchema>>({
    resolver: zodResolver(appointmentTypeSchema),
    defaultValues: {
      url: "",
      title: "",
      duration: "",
      description: "",
      videoCallSoftware: "",
    },
  });

  const { data } = api.event.getEventType.useQuery(
    { eventId: id },
    {
      enabled: open,
    }
  );

  const onSubmit = async (values: z.infer<typeof appointmentTypeSchema>) => {
    try {
      const response = await updateEventType(id, values);

      if (response.success) {
        toast.success("Event updated successfully!");
        form.reset();
        setOpen(false);
      } else {
        toast.error("Something went wrong!");
      }
    } catch (error) {
      toast.error("Something went wrong!");
    }
  };

  useEffect(() => {
    if (data) {
      form.reset({
        url: data.url,
        title: data.title,
        description: data.description,
        duration: data.duration.toString(),
        videoCallSoftware: data.videoCallSoftware,
      });
    }
  }, [data, form]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Edit Event Type</DialogTitle>
          <DialogDescription>
            Make changes to your event type here. Click save when you're done.
          </DialogDescription>
        </DialogHeader>
        <div className="flex w-full items-center justify-center gap-5">
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="grid w-full grid-cols-2 items-start justify-start gap-5"
            >
              <FormField
                control={form.control}
                name="title"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Title</FormLabel>
                    <FormControl>
                      <Input placeholder="30 minute meeting" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="col-span-1 flex w-full flex-col items-center justify-center gap-3 pt-1">
                <Label
                  htmlFor="url-slug"
                  className="w-full text-left text-sm font-medium leading-none"
                >
                  URL Slug
                </Label>
                <div className="flex w-full items-center justify-center overflow-hidden rounded-md border border-input shadow-sm">
                  <span className="flex h-[36px] flex-1 items-center bg-gray-200 px-3 text-left text-sm dark:bg-sidebar">
                    {new URL(env.NEXT_PUBLIC_APP_URL).host}/
                  </span>
                  <FormField
                    control={form.control}
                    name="url"
                    render={({ field }) => (
                      <FormItem>
                        <FormControl>
                          <Input
                            placeholder="example-url"
                            className="rounded-none border-none bg-transparent shadow-none"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>
              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem className="col-span-2 w-full">
                    <FormLabel>Description</FormLabel>
                    <FormControl>
                      <Textarea
                        rows={5}
                        {...field}
                        className="resize-none"
                        placeholder="Discuss the Feasibility of the MVP Business Model."
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="duration"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Duration</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Duration" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="15">15 Minutes</SelectItem>
                        <SelectItem value="30">30 Minutes</SelectItem>
                        <SelectItem value="45">45 Minutes</SelectItem>
                        <SelectItem value="60">1 Hour</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="videoCallSoftware"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Video Call Software</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Video Call Software" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="Zoom Meeting">
                          Zoom Meeting
                        </SelectItem>
                        <SelectItem value="Microsoft Teams">
                          Microsoft Teams
                        </SelectItem>
                        <SelectItem value="Google Meet">Google Meet</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="col-span-2 flex w-full items-center justify-end">
                <Button
                  type="submit"
                  disabled={form.formState.isSubmitting}
                  variant="default"
                >
                  {form.formState.isSubmitting ? (
                    <>
                      <Loader2 className="animate-spin" /> Please Wait...
                    </>
                  ) : (
                    "Save Changes"
                  )}
                </Button>
              </div>
            </form>
          </Form>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default EditEventModal;
