"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2 } from "lucide-react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { env } from "@/env";
import { appointmentTypeSchema } from "@/lib/validators";
import { api } from "@/trpc/react";

const Page = () => {
  const form = useForm<z.infer<typeof appointmentTypeSchema>>({
    resolver: zodResolver(appointmentTypeSchema),
    defaultValues: {
      url: "",
      title: "",
      duration: "15",
      description: "",
      videoCallSoftware: "",
    },
  });
  const createEvent = api.event.createEventType.useMutation();

  const onSubmit = async (values: z.infer<typeof appointmentTypeSchema>) => {
    const response = await createEvent.mutateAsync({
      url: values.url,
      title: values.title,
      description: values.description,
      duration: Number(values.duration),
      videoCallSoftware: values.videoCallSoftware,
    });

    if (response) {
      toast.success("Event type created successfully!");
    } else {
      toast.error("Something went wrong!");
    }
  };

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-5 p-5">
      <Card className="flex h-full w-1/2 flex-col">
        <CardHeader className="w-full rounded-t-lg border-b bg-sidebar">
          <CardTitle className="text-2xl text-primary">
            Appointment Types
          </CardTitle>
          <CardDescription>
            Create appointment types that allow people to book you.
          </CardDescription>
        </CardHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="w-full">
            <CardContent className="grid w-full grid-cols-2 items-start justify-start gap-5 p-5">
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
              <div className="col-span-1 flex w-full flex-col items-center justify-center gap-2">
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
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={`${field.value}`}
                    >
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
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                    >
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
            </CardContent>
            <div className="flex w-full items-center justify-end px-5">
              <Button
                type="submit"
                disabled={createEvent.isPending}
                variant="default"
              >
                {createEvent.isPending ? (
                  <>
                    <Loader2 className="animate-spin" /> Please Wait...
                  </>
                ) : (
                  "Create Appointment Type"
                )}
              </Button>
            </div>
          </form>
        </Form>
      </Card>
    </div>
  );
};

export default Page;
