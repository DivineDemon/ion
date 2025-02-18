import { redirect } from "next/navigation";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";

import NotFound from "@/components/not-found";
import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

const Page = async () => {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    redirect("/sign-in");
  }

  const dbUser = await db.user.findUnique({
    where: {
      id: user.id,
    },
    select: {
      grantId: true,
      grantEmail: true,
    },
  });

  if (!dbUser) {
    throw new Error("User not found!");
  }

  const data = await nylas.events.list({
    identifier: dbUser.grantId[0] as string,
    queryParams: {
      calendarId: dbUser.grantEmail[0] as string,
    },
  });

  return (
    <div className="flex h-screen w-full items-center justify-center">
      <NotFound title="No Meetings Found!" description="scheduling a meeting" />
    </div>
  );
};

export default Page;
