"use client";

import { useQuery } from "@tanstack/react-query";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { client } from "@/lib/client";

const Page = () => {
  const { data } = useQuery({
    queryFn: async () => {
      const response = await client.user.getDatabaseUser.$get({});
      return await response.json();
    },
    queryKey: ["get-database-user"],
  });

  console.log(data);

  return (
    <div className="flex w-1/2 flex-col items-start justify-start gap-5 divide-y divide-gray-300 rounded-xl border bg-white p-5">
      <div className="flex w-full flex-col items-center justify-center">
        <span className="w-full text-left text-2xl font-semibold text-primary">
          Profile
        </span>
        <span className="w-full text-left text-sm text-gray-400">
          Update or View your Information Here.
        </span>
      </div>
      <div className="flex w-full flex-col items-center justify-center gap-1 pt-5">
        <Label
          htmlFor="username"
          className="w-full text-left text-sm text-gray-400"
        >
          Username
        </Label>
        <Input
          type="text"
          placeholder={data?.user.username}
          className="w-full"
        />
      </div>
      <div className="flex w-full flex-col items-center justify-center gap-1 pt-5">
        <Label
          htmlFor="email"
          className="w-full text-left text-sm text-gray-400"
        >
          Email
        </Label>
        <Input type="text" placeholder={data?.user.email} className="w-full" />
      </div>
    </div>
  );
};

export default Page;
