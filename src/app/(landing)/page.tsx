import { Check } from "lucide-react";

import Heading from "@/components/heading";
import MaxWidthWrapper from "@/components/max-width-wrapper";
import ShinyButton from "@/components/shiny-button";

const App = () => {
  return (
    <>
      <section className="relative bg-gray-50 py-24 sm:py-32">
        <MaxWidthWrapper className="text-center">
          <div className="relative mx-auto flex flex-col items-center gap-10 text-center">
            <div className="">
              <Heading>
                <span>Real-Time SaaS Insights,</span>
                <br />
                <span className="relative bg-gradient-to-r from-primary to-red-800 bg-clip-text text-transparent">
                  Delivered to your Discord
                </span>
              </Heading>
            </div>
            <p className="max-w-prose text-pretty text-center text-base/7 text-gray-600">
              PingPanda is the easiest way to monitor your SaaS. Get instant
              notifications for&nbsp;
              <span className="font-semibold text-gray-700">
                sales, new users, or any other event
              </span>
              &nbsp;sent directly to your Discord
            </p>
            <ul className="flex flex-col items-start space-y-2 text-left text-base/7 text-gray-600">
              {[
                "Real-Time Discord alerts for critical events",
                "Buy once, use forever",
                "Track sales, new users, or any other event",
              ].map((item, idx) => (
                <li key={idx} className="flex items-center gap-1.5 text-left">
                  <Check className="size-5 shrink-0 text-primary" />
                  {item}
                </li>
              ))}
            </ul>
            <div className="w-full max-w-80">
              <ShinyButton
                href="/sign-up"
                className="relative z-10 h-14 w-full text-base shadow-lg transition-shadow duration-300 hover:shadow-xl"
              >
                Start for Free Today
              </ShinyButton>
            </div>
          </div>
        </MaxWidthWrapper>
      </section>
    </>
  );
};

export default App;
