"use client";

import { Loader2 } from "lucide-react";
import { useFormStatus } from "react-dom";

import { Button } from "./ui/button";

const SubmitButton = ({
  text,
  className,
}: {
  text: string;
  className?: string;
}) => {
  const { pending } = useFormStatus();

  return pending ? (
    <Button
      type="button"
      variant="default"
      disabled={true}
      className={className}
    >
      <Loader2 className="animate-spin" />
      <span>Please Wait...</span>
    </Button>
  ) : (
    <Button type="submit" variant="default" className={className}>
      {text}
    </Button>
  );
};

export default SubmitButton;
