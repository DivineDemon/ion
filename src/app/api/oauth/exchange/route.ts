import { redirect } from "next/navigation";
import { NextRequest } from "next/server";

import { currentUser } from "@clerk/nextjs/server";

import { db } from "@/db";
import { env } from "@/env";
import { nylas } from "@/lib/nylas";

export async function GET(req: NextRequest) {
  const auth = await currentUser();
  const url = new URL(req.url);
  const code = url.searchParams.get("code");

  if (!code) {
    return Response.json(
      {
        error: "No authorization code returned from Nylas",
      },
      { status: 400 }
    );
  }

  try {
    const response = await nylas.auth.exchangeCodeForToken({
      clientSecret: env.NYLAS_API_KEY,
      clientId: env.NYLAS_CLIENT_ID,
      code,
      redirectUri: "http://localhost:3000/api/oauth/exchange",
    });

    const { grantId, email } = response;
    await db.user.update({
      where: {
        externalId: auth?.id,
      },
      data: {
        grantId,
        grantEmail: email,
      },
    });
  } catch (error: Error | unknown) {
    throw new Error((error as Error).message);
  }

  redirect("/dashboard");
}
