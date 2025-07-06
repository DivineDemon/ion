"use client";

import { Loader2 } from "lucide-react";
import { useFormStatus } from "react-dom";

import { Button } from "./ui/button";

const SubmitButton = ({
  text,
  variant,
  className,
}: {
  text: string;
  className?: string;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link" | null | undefined;
}) => {
  const { pending } = useFormStatus();

  return pending ? (
    <Button type="button" disabled={true} variant={variant} className={className}>
      <Loader2 className="animate-spin" />
      <span>Please Wait...</span>
    </Button>
  ) : (
    <Button type="submit" variant={variant} className={className}>
      {text}
    </Button>
  );
};

export default SubmitButton;
