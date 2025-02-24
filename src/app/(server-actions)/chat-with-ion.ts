"use server";

import { type ChatCompletionMessageParam } from "openai/resources/index.mjs";

import { deleteEvent, getAllEvents } from "@/app/(server-actions)/ai-tools";
import { openai } from "@/lib/ai";
import { SYSTEM_PROMPT } from "@/lib/constants";

export async function chatWithIon(
  query: string,
  chatHistory: ChatCompletionMessageParam[]
) {
  let messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: SYSTEM_PROMPT,
    },
    {
      role: "user",
      content: query,
    },
  ];

  const initialResponse = await openai.chat.completions.create({
    model: "gpt-4-turbo-2024-04-09",
    messages: [...messages, ...chatHistory],
    tools: [
      {
        type: "function",
        function: {
          name: "get_events_summary",
          description: "Summarizes the user's event schedule.",
          parameters: {
            type: "object",
            properties: {
              grantEmail: {
                type: "string",
                description:
                  "The Email that is registered with nylas and against which the user requires the event schedule summary.",
              },
            },
            required: ["grantEmail"],
          },
        },
      },
      {
        type: "function",
        function: {
          name: "cancel_event",
          description: "Cancels an event.",
          parameters: {
            type: "object",
            properties: {
              eventId: {
                type: "string",
                description: "The ID of the event to be cancelled.",
              },
              grantEmail: {
                type: "string",
                description:
                  "The Email that is registered with nylas and against which the user requires the event schedule summary.",
              },
            },
            required: ["eventId", "grantEmail"],
          },
        },
      },
    ],
  });

  if (initialResponse.choices[0]?.message.tool_calls) {
    const toolCall = initialResponse.choices[0]?.message.tool_calls[0];

    if (toolCall?.function.name === "get_events_summary") {
      const userEmail: { grantEmail: string } = JSON.parse(
        toolCall.function.arguments
      );

      const summary = await getAllEvents(userEmail.grantEmail);

      messages.push({
        role: "assistant",
        content: null,
        tool_calls: [toolCall],
      });

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(summary),
      });

      const finalResponse = await openai.chat.completions.create({
        model: "gpt-4-turbo-2024-04-09",
        messages: [...messages, ...chatHistory],
        stream: true,
      });

      return finalResponse.toReadableStream();
    }

    if (toolCall?.function.name === "cancel_event") {
      const eventData: { eventId: string; grantEmail: string } = JSON.parse(
        toolCall.function.arguments
      );

      const response = await deleteEvent(
        eventData.grantEmail,
        eventData.eventId
      );

      messages.push({
        role: "assistant",
        content: null,
        tool_calls: [toolCall],
      });

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: JSON.stringify(response),
      });

      const finalResponse = await openai.chat.completions.create({
        model: "gpt-4-turbo-2024-04-09",
        messages: [...messages, ...chatHistory],
        stream: true,
      });

      return finalResponse.toReadableStream();
    }
  }

  const streamResponse = await openai.chat.completions.create({
    model: "gpt-4-turbo-2024-04-09",
    messages: [...messages, ...chatHistory],
    stream: true,
  });

  return streamResponse.toReadableStream();
}
