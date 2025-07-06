import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { notFound, redirect } from "next/navigation";

import EventCard from "@/components/event/event-card";
import EventForm from "@/components/event/event-form";
import NotFound from "@/components/not-found";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { db } from "@/server/db";

const Page = async () => {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return redirect("/sign-in");
  }

  const data = await db.eventType.findMany({
    where: {
      userId: user.id,
    },
  });

  if (!data) {
    return notFound();
  }

  return (
    <div className="flex h-full w-full items-start justify-start gap-5 p-5">
      <Card className="flex h-full w-1/2 flex-col">
        <CardHeader className="w-full rounded-t-lg border-b bg-sidebar">
          <CardTitle className="text-2xl text-primary">Appointment Types</CardTitle>
          <CardDescription>Create appointment types that allow people to book you.</CardDescription>
        </CardHeader>
        <EventForm />
      </Card>
      <div
        className={cn("grid h-full max-h-full flex-1 grid-cols-2 gap-5 overflow-y-auto", {
          "h-fit": data,
        })}
      >
        {!data ? (
          <div className="col-span-2 flex h-full w-full items-center justify-center">
            <NotFound title="No Events Found!" description="creating an event." />
          </div>
        ) : (
          data?.map((event) => <EventCard key={event.id} data={event} />)
        )}
      </div>
    </div>
  );
};

export default Page;
