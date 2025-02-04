import { string, z } from "zod";

export const userSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  lastName: z.string().optional(),
  imageUrl: z.string().optional(),
  firstName: z.string().optional(),
});
