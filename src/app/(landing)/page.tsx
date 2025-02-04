import Image from "next/image";

import { Atom, Check, Star, Verified } from "lucide-react";

import Feature from "@/assets/img/feature.png";
import Invite from "@/assets/img/invite.svg";
import MeetingManagement from "@/assets/img/meeting.svg";
import Questions from "@/assets/img/questions.svg";
import Track from "@/assets/img/track.svg";
import Heading from "@/components/heading";
import MaxWidthWrapper from "@/components/max-width-wrapper";
import ShinyButton from "@/components/shiny-button";
import BackgroundPattern from "@/components/ui/background-pattern";

const App = () => {
  return (
    <>
      <section className="relative mt-16 flex h-[calc(100vh-64px)] flex-col items-center justify-center">
        <BackgroundPattern className="absolute left-1/2 top-20 z-0 -translate-x-1/2 opacity-35" />
        <div className="absolute left-1/2 top-1/2 z-0 size-72 rounded-full bg-primary/50 blur-[50px]" />
        <div className="absolute left-1/3 top-1/3 z-0 size-36 rounded-full bg-green-500/50 blur-[50px]" />
        <div className="relative z-[1] flex h-full flex-col items-center justify-center gap-6 pb-16 text-center">
          <Atom className="size-12 text-primary" />
          <Heading className="bg-gradient-to-r from-yellow-900 via-primary to-yellow-400 bg-clip-text text-transparent">
            Collaboration, now a Breeze
          </Heading>
          <p className="max-w-prose text-base/7 text-gray-600">
            With <span className="font-bold text-primary">Ion</span>, you can
            achieve with your team what you couldn't
            <br />
            before. Next gen team collaboration with an intuitive interface.
          </p>
          <ul className="flex flex-col items-start space-y-2 text-left text-base/7 text-gray-600">
            {[
              "Real-Time Code repository analysis.",
              "Repository Commits analysis.",
              "Seamless Team Collaboration.",
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
      </section>
      <section className="relative pb-4">
        <div className="absolute inset-x-0 bottom-24 top-24 bg-primary" />
        <div className="relative mx-auto">
          <MaxWidthWrapper className="relative">
            <div className="-m-2 rounded-xl bg-gray-900/5 p-2 ring-1 ring-inset ring-gray-900/10 lg:-m-4 lg:rounded-2xl lg:p-4">
              <Image
                src={Feature}
                alt="feature-project"
                className="rounded-lg"
              />
            </div>
          </MaxWidthWrapper>
        </div>
      </section>
      <section className="relative bg-gray-50 py-24 sm:py-32">
        <MaxWidthWrapper className="flex flex-col items-center gap-16 sm:gap-20">
          <div className="">
            <h2 className="text-center text-base/7 font-semibold text-primary">
              Intuitive Collaboration
            </h2>
            <Heading>Stay Ahead with Modern Team Management</Heading>
          </div>
          <div className="grid gap-4 lg:grid-cols-3 lg:grid-rows-2">
            <div className="relative lg:row-span-2">
              <div className="absolute inset-px rounded-lg bg-white lg:rounded-l-[2rem]" />
              <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)] lg:rounded-l-[calc(2rem+1px)]">
                <div className="px-8 pb-3 pt-8 sm:px-10 sm:pb-0 sm:pt-10">
                  <p className="mt-2 text-lg/7 font-medium tracking-tight text-yellow-950 max-lg:text-center">
                    Meeting Management
                  </p>
                  <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
                    Harness the power of AI to summarize and track meetings,
                    extract and list issues discussed.
                  </p>
                </div>
                <div className="relative min-h-[30rem] w-full grow p-7 [container-type:inline-size] max-lg:mx-auto max-lg:max-w-sm">
                  <div className="absolute -bottom-16 -right-16 z-0 size-72 rounded-full bg-primary/50 blur-[50px]" />
                  <div className="absolute left-0 top-10 z-0 size-36 rounded-full bg-green-500/50 blur-[50px]" />
                  <Image
                    src={MeetingManagement}
                    alt="meeting-management"
                    className="size-full object-contain"
                  />
                </div>
              </div>
              <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5 lg:rounded-l-[2rem]" />
            </div>
            <div className="relative max-lg:row-start-1">
              <div className="absolute inset-px rounded-lg bg-white max-lg:rounded-t-[2rem]" />
              <div className="relative flex h-full flex-col gap-5 overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)] max-lg:rounded-t-[calc(2rem+1px)]">
                <div className="px-8 pt-8 sm:px-10 sm:pt-10">
                  <p className="mt-2 text-lg/7 font-medium tracking-tight text-yellow-950 max-lg:text-center">
                    Ask Questions
                  </p>
                  <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
                    Ask the Ion AI questions about your code and find answers to
                    your questions.
                  </p>
                </div>
                <div className="flex flex-1 items-center justify-center px-8 max-lg:pb-12 max-lg:pt-10 sm:px-10 lg:pb-2">
                  <Image
                    src={Questions}
                    alt="questions"
                    width={500}
                    height={300}
                    className="w-full max-lg:max-w-xs"
                  />
                </div>
              </div>
              <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5 max-lg:rounded-t-[2rem]" />
            </div>
            <div className="relative max-lg:row-start-3 lg:col-start-2 lg:row-start-2">
              <div className="absolute inset-px rounded-lg bg-white" />
              <div className="relative flex h-full flex-col gap-5 overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)]">
                <div className="px-8 pt-8 sm:px-10 sm:pt-10">
                  <p className="mt-2 text-lg/7 font-medium tracking-tight text-yellow-950 max-lg:text-center">
                    Invite Members
                  </p>
                  <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
                    Onboard new members onto your team with ease. Employ Ion AI
                    to introduce your team members to your codebase.
                  </p>
                </div>
                <div className="flex flex-1 items-center justify-center px-8 max-lg:pb-12 max-lg:pt-10 sm:px-10 lg:pb-2">
                  <Image
                    src={Invite}
                    alt="invite"
                    width={500}
                    height={300}
                    className="w-full max-lg:max-w-xs"
                  />
                </div>
              </div>
              <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5" />
            </div>
            <div className="relative lg:row-span-2">
              <div className="absolute inset-px rounded-lg bg-white max-lg:rounded-b-[2rem] lg:rounded-r-[2rem]" />
              <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(theme(borderRadius.lg)+1px)] max-lg:rounded-b-[calc(2rem+1px)] lg:rounded-r-[calc(2rem+1px)]">
                <div className="px-8 pb-3 pt-8 sm:px-10 sm:pb-0 sm:pt-10">
                  <p className="mt-2 text-lg/7 font-medium tracking-tight text-yellow-950 max-lg:text-center">
                    Track commits
                  </p>
                  <p className="mt-2 max-w-lg text-sm/6 text-gray-600 max-lg:text-center">
                    Keep track of the changes made to your codebase with ease.
                    Get summarized, easy to read insights about your codebase.
                  </p>
                </div>
                <div className="relative min-h-[30rem] w-full grow p-7 [container-type:inline-size] max-lg:mx-auto max-lg:max-w-sm">
                  <div className="absolute -bottom-16 -left-16 z-0 size-72 rounded-full bg-primary/50 blur-[50px]" />
                  <div className="absolute right-0 top-10 z-0 size-36 rounded-full bg-green-500/50 blur-[50px]" />
                  <Image
                    src={Track}
                    alt="track"
                    className="size-full object-contain"
                  />
                </div>
              </div>
              <div className="pointer-events-none absolute inset-px rounded-lg shadow ring-1 ring-black/5 max-lg:rounded-b-[2rem] lg:rounded-r-[2rem]" />
            </div>
          </div>
        </MaxWidthWrapper>
      </section>
      <section className="relative bg-white py-24 sm:py-32">
        <MaxWidthWrapper className="flex flex-col items-center gap-16 sm:gap-20">
          <div className="">
            <h2 className="text-center text-base/7 font-semibold text-primary">
              Real-World Experiences
            </h2>
            <Heading className="text-center">What our Customers Say</Heading>
          </div>
          <div className="mx-auto grid max-w-2xl grid-cols-1 divide-y divide-gray-200 px-4 lg:mx-0 lg:max-w-none lg:grid-cols-2 lg:divide-x lg:divide-y-0">
            <div className="flex flex-auto flex-col gap-4 rounded-t-[2rem] bg-gray-50 p-6 sm:p-8 lg:rounded-l-[2rem] lg:rounded-tr-none lg:p-16">
              <div className="lg:justify-star mb-2 flex justify-center gap-0.5">
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
              </div>
              <p className="text-pretty text-center text-base font-medium tracking-tight text-yellow-950 sm:text-lg lg:text-left lg:text-lg/8">
                Ion has been a gamechanger for me. I&apos;ve been using it for 2
                months now and seeing team performance increase in real-time is
                super satisfying.
              </p>
              <div className="mt-2 flex flex-col items-center justify-center gap-4 sm:flex-row sm:items-start lg:justify-start">
                <Image
                  src="https://ui.shadcn.com/avatars/01.png"
                  className="rounded-full object-cover"
                  alt="random-user"
                  width={48}
                  height={48}
                />
                <div className="flex flex-col items-center sm:items-start">
                  <p className="flex items-center font-semibold">
                    Freya Larsson
                    <Verified className="ml-1.5 size-5 fill-primary text-white" />
                  </p>
                  <p className="text-sm text-gray-600">@itsfreyaa</p>
                </div>
              </div>
            </div>
            <div className="flex flex-auto flex-col gap-4 rounded-b-[2rem] bg-gray-50 p-6 sm:p-8 lg:rounded-r-[2rem] lg:rounded-bl-none lg:p-16">
              <div className="lg:justify-star mb-2 flex justify-center gap-0.5">
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
                <Star className="size-5 fill-primary text-primary" />
              </div>
              <p className="text-pretty text-center text-base font-medium tracking-tight text-yellow-950 sm:text-lg lg:text-left lg:text-lg/8">
                Ion has been paying off for our team. Nice to have a simple way
                of seeing how we&apos;re doing day-to-day. Definitely makes our
                lives easier.
              </p>
              <div className="mt-2 flex flex-col items-center justify-center gap-4 sm:flex-row sm:items-start lg:justify-start">
                <Image
                  src="https://ui.shadcn.com/avatars/04.png"
                  className="rounded-full object-cover"
                  alt="random-user"
                  width={48}
                  height={48}
                />
                <div className="flex flex-col items-center sm:items-start">
                  <p className="flex items-center font-semibold">
                    Kai Durant
                    <Verified className="ml-1.5 size-5 fill-primary text-white" />
                  </p>
                  <p className="text-sm text-gray-600">@kaidurant_</p>
                </div>
              </div>
            </div>
          </div>
          <ShinyButton
            href="/sign-up"
            className="relative z-10 h-14 w-full max-w-xs text-base shadow-lg transition-shadow duration-300 hover:shadow-xl"
          >
            Start For Free Today
          </ShinyButton>
        </MaxWidthWrapper>
      </section>
    </>
  );
};

export default App;
