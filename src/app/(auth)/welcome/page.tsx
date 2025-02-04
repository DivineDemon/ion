import { notFound, redirect } from "next/navigation";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";

import Heading from "@/components/heading";
import LoadingSpinner from "@/components/loading-spinner";
import BackgroundPattern from "@/components/ui/background-pattern";
import { api } from "@/trpc/server";

const Page = async () => {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return notFound();
  }

  const checkUser = await api.user.findUser();

  if (checkUser) {
    redirect("/dashboard");
  }

  const response = await api.user.syncUser({
    id: user.id,
    email: user.email!,
    imageUrl: user.picture!,
    firstName: user.given_name!,
    lastName: user.family_name!,
  });

  if (response) {
    redirect("/dashboard");
  }

  return (
    <div className="flex w-full flex-1 items-center justify-center p-4">
      <BackgroundPattern className="absolute inset-0 left-1/2 z-0 -translate-x-1/2 opacity-35" />
      <div className="relative z-10 flex -translate-y-1/2 flex-col items-center gap-6 text-center">
        <LoadingSpinner />
        <Heading>Creating your Account...</Heading>
        <p className="max-w-prose text-base/7 text-gray-600">
          Just a moment while we set things up for you.
        </p>
      </div>
    </div>
  );
};

export default Page;
