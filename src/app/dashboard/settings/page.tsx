"use client";

import Image from "next/image";
import Link from "next/link";
import { ChangeEvent, useEffect, useRef, useState } from "react";

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2, Trash } from "lucide-react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";

import { Button, buttonVariants } from "@/components/ui/button";
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
import { cn, parseImage } from "@/lib/utils";
import { api } from "@/trpc/react";

const userFormSchema = z.object({
  email: z.string(),
  lastName: z.string(),
  userName: z.string(),
  firstName: z.string(),
});

const Page = () => {
  const { data: linkedEmails, isLoading } =
    api.user.fetchUserLinkedAccounts.useQuery();
  const { data } = api.user.findUser.useQuery();
  const fileRef = useRef<HTMLInputElement>(null);
  const [image, setImage] = useState<string>("");
  const updateUser = api.user.updateUser.useMutation();
  const [uploading, setUploading] = useState<boolean>(false);

  const form = useForm<z.infer<typeof userFormSchema>>({
    resolver: zodResolver(userFormSchema),
    defaultValues: {
      email: "",
      lastName: "",
      userName: "",
      firstName: "",
    },
  });

  const triggerUpload = () => {
    if (fileRef.current) {
      fileRef.current.click();
    }
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    setUploading(true);

    if (e.target.files && e.target.files.length > 0) {
      let url = null;
      const file = e.target.files[0];

      if (file) {
        url = await parseImage(file);
      }

      setImage(`${url}`);
      setUploading(false);
    }
  };

  const onSubmit = async (values: z.infer<typeof userFormSchema>) => {
    const response = await updateUser.mutateAsync({
      imageUrl: image,
      email: values.email,
      lastName: values.lastName,
      firstName: values.firstName,
      userName: `${values?.userName}`,
    });

    if (response) {
      toast.success("User updated successfully!");
    } else {
      toast.error("Something went wrong!");
    }
  };

  useEffect(() => {
    if (data) {
      form.setValue("email", data?.email);
      form.setValue("lastName", `${data?.lastName}`);
      form.setValue("firstName", `${data?.firstName}`);
      form.setValue("userName", `${data?.userName ? data?.userName : ""}`);
    }
  }, [data]);

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-5 p-5">
      <Card className="flex h-full w-1/2 flex-col">
        <CardHeader className="w-full rounded-t-lg border-b bg-sidebar">
          <CardTitle className="text-2xl text-primary">
            Account Settings
          </CardTitle>
          <CardDescription>
            Update or view your account settings here.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="grid w-full grid-cols-2 gap-4 pt-5"
            >
              <div className="col-span-2 flex w-full items-center justify-start gap-4">
                <div className="flex size-[85px] items-center justify-center rounded-full bg-primary/20 p-2">
                  {data ? (
                    uploading ? (
                      <Loader2 className="size-10 animate-spin text-yellow-600" />
                    ) : (
                      <Image
                        src={`${image ? image : data?.imageUrl}`}
                        alt="user-dp"
                        width={85}
                        height={85}
                        className="size-full rounded-full"
                      />
                    )
                  ) : null}
                </div>
                <div className="flex flex-1 flex-col items-start justify-center gap-2">
                  <span className="w-full text-left text-[18px] font-semibold leading-[18px]">
                    Upload Image
                  </span>
                  <span className="w-full text-left text-[12px] leading-[12px] text-gray-500">
                    Accepted Formats: .png, .jpeg, .jpg
                  </span>
                  <input
                    ref={fileRef}
                    type="file"
                    className="hidden"
                    multiple={false}
                    onChange={handleFileUpload}
                    accept="image/png, image/jpeg, image/jpg"
                  />
                  <Button
                    type="button"
                    onClick={triggerUpload}
                    variant="outline"
                    size="sm"
                  >
                    Choose Image
                  </Button>
                </div>
              </div>
              <FormField
                control={form.control}
                name="firstName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>First Name</FormLabel>
                    <FormControl>
                      <Input placeholder="John" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="lastName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Last Name</FormLabel>
                    <FormControl>
                      <Input placeholder="Doe" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email</FormLabel>
                    <FormControl>
                      <Input
                        type="email"
                        placeholder="johndoe@email.com"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="userName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Username</FormLabel>
                    <FormControl>
                      <Input placeholder="johndoe123" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="col-span-2 flex w-full items-center justify-end gap-4">
                <Button
                  variant="default"
                  type="submit"
                  disabled={uploading || updateUser.isPending}
                >
                  {updateUser.isPending ? (
                    <Loader2 className="size-10 animate-spin text-black" />
                  ) : (
                    "Save"
                  )}
                </Button>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>
      <Card className="flex h-full w-1/2 flex-col">
        <CardHeader className="w-full rounded-t-lg border-b bg-sidebar">
          <CardTitle className="text-2xl text-primary">
            Linked Accounts
          </CardTitle>
          <CardDescription>
            Add or Remove your Integrated accounts here.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex h-full max-h-full w-full flex-col items-start justify-start gap-5 overflow-y-auto p-5">
          {isLoading ? (
            <div className="flex h-full w-full items-center justify-center">
              <Loader2 className="size-10 animate-spin text-primary" />
            </div>
          ) : (
            linkedEmails?.grantEmail?.map((account, idx) => (
              <div
                key={idx}
                className="flex w-full items-center justify-center gap-4"
              >
                <Input
                  type="email"
                  placeholder={account}
                  disabled
                  className="flex-1"
                />
                <Button
                  type="button"
                  className="shrink-0"
                  variant="destructive"
                  size="icon"
                >
                  <Trash />
                </Button>
              </div>
            ))
          )}
          <div className="flex w-full items-center justify-end">
            <Link
              href="/api/auth"
              className="rounded-md bg-primary px-5 py-2 text-sm font-medium text-black"
            >
              Add Account
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Page;
