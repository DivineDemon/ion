import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

const spinnerVariants = cva(
  "border-4 rounded-full border-primary/20 border-t-primary/70 animate-spin duration-700",
  {
    variants: {
      size: {
        sm: "size-4 border-2",
        md: "size-6 border-4",
        lg: "size-8 border-4",
      },
    },
    defaultVariants: {
      size: "md",
    },
  }
);

const LoadingSpinner = () => {
  return (
    <div className="flex items-center justify-center">
      <div
        className={cn(spinnerVariants({ size: "md", className: "h-8 w-8" }))}
      />
    </div>
  );
};

export default LoadingSpinner;
