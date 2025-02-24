"use client";

import { type Dispatch, type SetStateAction, useState } from "react";

import MDEditor from "@uiw/react-md-editor";
import { Send } from "lucide-react";
import { toast } from "sonner";

import { chatWithIon } from "@/app/(server-actions)/chat-with-ion";
import useAccount from "@/hooks/use-account";
import { cn } from "@/lib/utils";

import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Sheet, SheetContent } from "./ui/sheet";

interface ChatBarProps {
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
}

const ChatBar = ({ open, setOpen }: ChatBarProps) => {
  const { account } = useAccount();
  const [query, setQuery] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);

  const handleChat = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { id: prev.length + 1, type: "client", content: query },
      { id: prev.length + 2, type: "bot", content: "" },
    ]);
    setQuery("");

    try {
      const response: ReadableStream = await chatWithIon(
        `${query}. Email: ${account}`
      );

      if (!response) {
        toast.error("No response body received");
        return;
      }

      const reader = response.getReader();
      const decoder = new TextDecoder();
      let botMessage = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        const jsonObjects = chunk
          .split("\n")
          .filter((line) => line.trim())
          .map((line) => {
            try {
              return JSON.parse(line);
            } catch (e) {
              console.error("JSON Parse Error:", e);
              return null;
            }
          })
          .filter((json) => json !== null);

        for (const jsonObj of jsonObjects) {
          let content = "";

          if (typeof jsonObj === "string") {
            content = jsonObj;
          } else if (jsonObj?.choices?.[0]?.delta?.content) {
            const deltaContent = jsonObj.choices[0].delta.content;

            if (typeof deltaContent === "string") {
              content = deltaContent;
            } else if (
              typeof deltaContent === "object" &&
              deltaContent?.response
            ) {
              content = deltaContent.response;
            }
          } else if (jsonObj?.response) {
            content = jsonObj.response;
          }

          if (content) {
            botMessage += content;
            setMessages((prev) =>
              prev.map((msg, index) =>
                index === prev.length - 1
                  ? { ...msg, content: botMessage }
                  : msg
              )
            );
          }
        }
      }
    } catch (error) {
      toast.error("Chat Failed!");
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length + 1,
          type: "bot",
          content: "Error processing request.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent className="gap-0 space-y-0 p-0">
        <div className="flex h-full w-full flex-col items-center justify-between p-5">
          <div className="flex h-[calc(100vh-100px)] w-full flex-col gap-3 overflow-y-auto overflow-x-hidden rounded-xl border p-3">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex w-full ${
                  message.type === "client" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={cn(
                    "max-w-2/3 flex flex-wrap items-start justify-start break-all rounded-xl px-4 py-2 text-sm",
                    {
                      "bg-primary/10 text-right text-yellow-600":
                        message.type === "client",
                      "bg-muted text-left": message.type === "bot",
                    }
                  )}
                >
                  {message.content ? (
                    <MDEditor.Markdown
                      source={message.content}
                      style={{
                        background: "transparent",
                        padding: 0,
                      }}
                    />
                  ) : (
                    <div className="flex items-center gap-1">
                      <div className="size-2 animate-fadeDots rounded-full bg-primary" />
                      <div
                        className="size-2 animate-fadeDots rounded-full bg-primary"
                        style={{ animationDelay: "0.2s" }}
                      />
                      <div
                        className="size-2 animate-fadeDots rounded-full bg-primary"
                        style={{ animationDelay: "0.4s" }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleChat();
            }}
            className="flex w-full items-center justify-center gap-3"
          >
            <Input
              type="text"
              value={query}
              className="flex-1"
              disabled={loading}
              placeholder="What can I help you with?"
              onChange={(e) => setQuery(e.target.value)}
            />
            <Button disabled={loading} type="submit" variant="default">
              <Send />
            </Button>
          </form>
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default ChatBar;
