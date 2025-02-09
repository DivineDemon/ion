import { redirect } from "next/navigation";
import { NextRequest, NextResponse } from "next/server";

import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";

import { env } from "@/env";
import { nylas } from "@/lib/nylas";
import { db } from "@/server/db";

export async function GET(req: NextRequest) {
  const { getUser } = getKindeServerSession();
  const user = await getUser();

  if (!user) {
    return NextResponse.json({ message: "Unauthorized" }, { status: 401 });
  }

  const url = new URL(req.url);
  const code = url.searchParams.get("code");

  if (!code) {
    return NextResponse.json({ message: "Missing code!" }, { status: 400 });
  }

  try {
    const response = await nylas.auth.exchangeCodeForToken({
      clientSecret: env.NYLAS_API_SECRET_KEY,
      clientId: env.NYLAS_CLIENT_ID,
      redirectUri: `${env.NEXT_PUBLIC_APP_URL}/api/oauth/exchange`,
      code,
    });

    const { grantId, email } = response;

    await db.user.update({
      where: {
        id: user.id,
      },
      data: {
        grantId: {
          push: grantId,
        },
        grantEmail: {
          push: email,
        },
      },
    });
  } catch (error: Error | unknown) {
    return NextResponse.json(
      { message: "Error exchanging code for token", error },
      { status: 500 }
    );
  }

  redirect("/dashboard");
}
