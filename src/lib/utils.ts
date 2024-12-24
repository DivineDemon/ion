import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

import { env } from "@/env";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export async function base64ToUrl(base64: string): Promise<string | undefined> {
  const formData = new FormData();
  formData.append("image", base64);

  const response = await fetch(
    `https://api.imgbb.com/1/upload?key=${env.NEXT_PUBLIC_IMGBB_API_KEY}`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (response.ok) {
    const data = await response.json();
    return data.data.url;
  } else {
    return undefined;
  }
}

export async function fileToBase64(file: File): Promise<string | undefined> {
  const base64 = await new Promise<string | undefined>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result) {
        resolve(reader.result.toString());
      } else {
        resolve(undefined);
      }
    };
    reader.onerror = () => reject(new Error("Error reading file as base64"));
    reader.readAsDataURL(file);
  });

  if (!base64) {
    return undefined;
  }

  const imageUrl = await base64ToUrl(base64.split(",")[1]);
  return imageUrl;
}
