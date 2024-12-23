import Nylas from "nylas";

import { env } from "@/env";

export const nylas = new Nylas({
  apiKey: env.NYLAS_API_KEY,
  apiUri: env.NYLAS_API_URL,
});
