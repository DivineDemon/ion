import Nylas from "nylas";

import { env } from "@/env";

export const nylas = new Nylas({
  apiKey: env.NYLAS_API_SECRET_KEY,
  apiUri: env.NYLAS_API_URI,
});
