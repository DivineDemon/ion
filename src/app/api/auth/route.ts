import { redirect } from "next/navigation";

import { env } from "@/env";
import { nylas } from "@/lib/nylas";

export async function GET() {
  const authUrl = nylas.auth.urlForOAuth2({
    clientId: env.NYLAS_CLIENT_ID,
    redirectUri: `${env.NEXT_PUBLIC_APP_URL}/api/oauth/exchange`,
  });

  return redirect(authUrl);
}
