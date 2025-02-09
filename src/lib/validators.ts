import { string, z } from "zod";

export const userSchema = z.object({
  lastName: z.string(),
  firstName: z.string(),
  email: z.string().email(),
  id: z.string().optional(),
  imageUrl: z.string().optional(),
  userName: z.string().optional(),
});
