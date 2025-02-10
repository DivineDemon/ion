import { notFound, redirect } from "next/navigation";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";

import { updateAvailability } from "@/app/(server-actions)/update-availability";
import SubmitButton from "@/components/submit-button";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { TIME_SLOTS } from "@/lib/constants";
import { db } from "@/server/db";

const Page = async () => {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    redirect("/sign-in");
  }

  const data = await db.availability.findMany({
    where: {
      userId: user.id,
    },
  });

  if (!data) {
    return notFound();
  }

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-5 p-5">
      <Card className="flex h-full w-1/2 flex-col">
        <CardHeader className="w-full rounded-t-lg border-b bg-sidebar">
          <CardTitle className="text-2xl text-primary">Availability</CardTitle>
          <CardDescription>
            Manage the days and hours you are available here.
          </CardDescription>
        </CardHeader>
        <form action={updateAvailability} className="w-full">
          <CardContent className="flex w-full flex-col items-start justify-start gap-5 p-0 py-5">
            {data.map((availability) => (
              <div
                key={availability.id}
                className="flex w-full items-center justify-center gap-4 px-5"
              >
                <input
                  type="hidden"
                  name={`id-${availability.id}`}
                  value={availability.id}
                />
                <div className="flex w-1/4 shrink-0 items-center justify-start gap-4">
                  <Switch
                    name={`isActive-${availability.id}`}
                    defaultChecked={availability.isActive}
                  />
                  <span>{availability.day}</span>
                </div>
                <div className="grid flex-1 grid-cols-2 gap-4">
                  <Select
                    name={`fromTime-${availability.id}`}
                    defaultValue={availability.fromTime}
                  >
                    <SelectTrigger className="col-span-1 w-full">
                      <SelectValue placeholder="From Time" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {TIME_SLOTS.map((slot) => (
                          <SelectItem key={slot.id} value={slot.time}>
                            {slot.time}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                  <Select
                    name={`tillTime-${availability.id}`}
                    defaultValue={availability.tillTime}
                  >
                    <SelectTrigger className="col-span-1 w-full">
                      <SelectValue placeholder="Till Time" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        {TIME_SLOTS.map((slot) => (
                          <SelectItem key={slot.id} value={slot.time}>
                            {slot.time}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            ))}
          </CardContent>
          <div className="flex w-full items-center justify-end px-5">
            <SubmitButton text="Save Changes" />
          </div>
        </form>
      </Card>
    </div>
  );
};

export default Page;
