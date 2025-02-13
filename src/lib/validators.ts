import { z } from "zod";

export const userSchema = z.object({
  lastName: z.string(),
  firstName: z.string(),
  email: z.string().email(),
  id: z.string().optional(),
  imageUrl: z.string().optional(),
  userName: z.string().optional(),
});

export const appointmentTypeSchema = z.object({
  title: z.string().min(3).max(150),
  duration: z.string(),
  url: z.string().min(3).max(150),
  description: z.string().min(3).max(300),
  videoCallSoftware: z.string().min(3),
});

export const postAppointmentSchema = z.object({
  url: z.string().min(3).max(150),
  title: z.string().min(3).max(150),
  duration: z.number().min(15).max(60),
  videoCallSoftware: z.string().min(3),
  description: z.string().min(3).max(300),
});

export const updateAppointmentSchema = z.object({
  id: z.string(),
  url: z.string().min(3).max(150),
  title: z.string().min(3).max(150),
  duration: z.number().min(15).max(60),
  videoCallSoftware: z.string().min(3),
  description: z.string().min(3).max(300),
});
