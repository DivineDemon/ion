import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

import { env } from "@/env";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export async function parseImage(file: File) {
  const formData = new FormData();
  formData.append("image", file);

  const converted = await fetch(
    `https://api.imgbb.com/1/upload?key=${env.NEXT_PUBLIC_IMGBB_KEY}`,
    {
      method: "POST",
      body: formData,
    }
  );

  const response: {
    data: { url: String };
  } = await converted.json();

  return response.data.url;
}
