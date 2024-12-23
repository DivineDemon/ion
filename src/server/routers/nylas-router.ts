import { env } from "@/env";
import { nylas } from "@/lib/nylas";

import { router } from "../__internals/router";
import { publicProcedure } from "../procedures";

export const nylasRouter = router({
  getRedirectUrl: publicProcedure.query(({ c }) => {
    const authUrl = nylas.auth.urlForOAuth2({
      clientId: env.NYLAS_CLIENT_ID,
      redirectUri: "http://localhost:3000/api/oauth/exchange",
    });

    return c.json({
      success: true,
      redirectUrl: authUrl,
    });
  }),
});
