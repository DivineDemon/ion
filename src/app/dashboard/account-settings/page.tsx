"use client";

import { useState } from "react";

import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import ImageUploader from "@/components/account-settings/image-uploader";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { client } from "@/lib/client";

const EditSchema = z.object({
  firstName: z
    .string()
    .min(1, "First name is required")
    .max(50, "First name must not exceed 50 characters"),
  lastName: z
    .string()
    .min(1, "Last name is required")
    .max(50, "Last name must not exceed 50 characters"),
  username: z
    .string()
    .min(3, "Username must be at least 3 characters")
    .max(20, "Username must not exceed 20 characters")
    .regex(
      /^[a-zA-Z0-9_]+$/,
      "Username can only contain letters, numbers, and underscores"
    ),
  email: z.string().email("Invalid email address"),
});

const Page = () => {
  const { mutate: updateUser, isPending } = useMutation({
    mutationFn: async (user: UserProps) => {
      const response = await client.user.updateUser.$post(user);
      return await response.json();
    },
  });
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryFn: async () => {
      const response = await client.user.getDatabaseUser.$get({});
      return await response.json();
    },
    queryKey: ["get-database-user"],
  });

  const form = useForm<z.infer<typeof EditSchema>>({
    resolver: zodResolver(EditSchema),
    defaultValues: {
      firstName: data?.user.firstName,
      lastName: data?.user.lastName,
      username: data?.user.username,
      email: data?.user.email,
    },
  });

  const onSubmit = (data: z.infer<typeof EditSchema>) => {
    updateUser({
      ...data,
      image: `${uploadedImage}`,
    });
  };

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="flex w-1/2 flex-col items-start justify-start gap-5 divide-y divide-gray-300 rounded-xl border bg-white p-5"
      >
        <div className="flex w-full flex-col items-center justify-center">
          <div className="flex w-full items-center justify-center text-primary">
            <span className="w-full text-left text-2xl font-semibold">
              Profile
            </span>
            {isLoading && <Loader2 className="animate-spin" />}
          </div>
          <span className="w-full text-left text-sm text-gray-400">
            Update or View your Information Here.
          </span>
        </div>
        <FormField
          control={form.control}
          name="username"
          render={({ field }) => (
            <FormItem className="flex w-full flex-col items-center justify-center gap-1 pt-5">
              <FormLabel className="w-full text-left text-sm text-gray-400">
                Username
              </FormLabel>
              <FormControl>
                <Input
                  placeholder={data?.user.username}
                  {...field}
                  className="w-full"
                />
              </FormControl>
              <FormMessage className="w-full text-left" />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="email"
          render={({ field }) => (
            <FormItem className="flex w-full flex-col items-center justify-center gap-1 pt-5">
              <FormLabel className="w-full text-left text-sm text-gray-400">
                Email
              </FormLabel>
              <FormControl>
                <Input
                  placeholder={data?.user.email}
                  {...field}
                  className="w-full"
                />
              </FormControl>
              <FormMessage className="w-full text-left" />
            </FormItem>
          )}
        />
        <div className="grid w-full grid-cols-2 gap-2.5 pt-5">
          <FormField
            control={form.control}
            name="firstName"
            render={({ field }) => (
              <FormItem className="flex w-full flex-col items-center justify-center gap-1">
                <FormLabel className="w-full text-left text-sm text-gray-400">
                  First Name
                </FormLabel>
                <FormControl>
                  <Input
                    placeholder={data?.user.firstName}
                    {...field}
                    className="w-full"
                  />
                </FormControl>
                <FormMessage className="w-full text-left" />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="lastName"
            render={({ field }) => (
              <FormItem className="flex w-full flex-col items-center justify-center gap-1">
                <FormLabel className="w-full text-left text-sm text-gray-400">
                  Last Name
                </FormLabel>
                <FormControl>
                  <Input
                    placeholder={data?.user.lastName}
                    {...field}
                    className="w-full"
                  />
                </FormControl>
                <FormMessage className="w-full text-left" />
              </FormItem>
            )}
          />
        </div>
        <div className="flex w-full items-center justify-center gap-2.5 pt-5">
          <ImageUploader
            userImage={`${data?.user.image}`}
            uploadedImage={uploadedImage}
            setUploadedImage={setUploadedImage}
          />
          <div className="flex flex-1 flex-col items-center justify-center">
            <span className="w-full text-left font-semibold">Upload Image</span>
            <span className="w-full text-left text-xs text-gray-400">
              Allowed formats: .jpg, .jpeg, .png
            </span>
            <span className="w-full text-left text-xs text-gray-400">
              Size less than 1MB
            </span>
          </div>
        </div>
        <div className="flex w-full flex-col items-center justify-center gap-1 pt-5">
          <span className="w-full text-left font-semibold">Linked Account</span>
          <div className="flex w-full items-center justify-center">
            <span className="flex-1 text-left text-sm text-primary">
              {data?.user.grantEmail}
            </span>
            <span className="bg-gradient-to-r from-pink-500 via-red-500 to-yellow-500 bg-clip-text text-sm font-bold text-transparent">
              Google
            </span>
          </div>
        </div>
        <Button disabled={isPending} type="submit" variant="default" size="lg">
          {isPending ? (
            <div className="flex w-full items-center justify-center gap-2.5">
              <Loader2 className="animate-spin" />
              <span>Please Wait...</span>
            </div>
          ) : (
            "Save Changes"
          )}
        </Button>
      </form>
    </Form>
  );
};

export default Page;
