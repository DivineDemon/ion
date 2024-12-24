import { cn } from "@/lib/utils";

interface CustomToastProps {
  title: string;
  description: string;
  type: "error" | "success" | "info";
}

const CustomToast = ({ type, title, description }: CustomToastProps) => {
  return (
    <div
      className={cn(
        "flex w-80 flex-col items-center justify-center gap-1 rounded-lg p-2.5",
        {
          "bg-primary/20 text-primary dark:bg-primary dark:text-white":
            type === "info",
          "bg-red-200 text-red-500 dark:bg-red-500 dark:text-white":
            type === "error",
          "bg-green-200 text-green-500 dark:bg-green-500 dark:text-white":
            type === "success",
        }
      )}
    >
      <span className="w-full text-left text-sm font-semibold">{title}</span>
      <span className="w-full text-left text-xs">{description}</span>
    </div>
  );
};

export default CustomToast;
