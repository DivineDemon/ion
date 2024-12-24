"use client";

import { Dispatch, SetStateAction, useCallback } from "react";

import { Camera } from "lucide-react";
import { useDropzone } from "react-dropzone";

import { fileToBase64 } from "@/lib/utils";

import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";

interface ImageUploaderProps {
  userImage: string;
  uploadedImage: string | null;
  setUploadedImage: Dispatch<SetStateAction<string | null>>;
}

const ImageUploader = ({
  userImage,
  uploadedImage,
  setUploadedImage,
}: ImageUploaderProps) => {
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const imageUrl = await fileToBase64(file);
      setUploadedImage(imageUrl || null);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      "image/*": [],
    },
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      className="group relative size-16 overflow-hidden rounded-full border"
    >
      <input {...getInputProps()} />
      <Avatar className="size-full">
        <AvatarImage src={uploadedImage || userImage} alt="user-profile" />
        <AvatarFallback>MH</AvatarFallback>
      </Avatar>
      <div className="absolute inset-0 hidden size-16 cursor-pointer items-center justify-center rounded-full bg-primary/50 shadow-none backdrop-blur-sm group-hover:flex">
        <Camera className="size-5 text-white" />
      </div>
    </div>
  );
};

export default ImageUploader;
