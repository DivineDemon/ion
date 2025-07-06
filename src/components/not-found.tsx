import Image from "next/image";

import NotFoundImage from "@/assets/img/not-found.svg";

interface NotFoundProps {
  title: string;
  description: string;
}

const NotFound = ({ title, description }: NotFoundProps) => {
  return (
    <div className="flex w-full flex-col items-center justify-center">
      <div className="relative flex size-80 items-center justify-center">
        <div className="z-0 size-80 rounded-full bg-primary/25 blur-[50px] filter" />
        <Image
          src={NotFoundImage}
          alt="not-found"
          width={288}
          height={288}
          className="absolute top-2.5 left-5 z-[1] size-72"
        />
      </div>
      <div className="flex w-full flex-col items-center justify-center gap-2">
        <span className="w-full text-center font-bold text-3xl text-yellow-700 capitalize">{title}</span>
        <span className="w-full text-center text-muted-foreground text-sm">
          No worries, enjoy your free time or
          <br />
          start by&nbsp;{description}.
        </span>
      </div>
    </div>
  );
};

export default NotFound;
