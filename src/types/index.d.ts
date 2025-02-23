declare type User = {
  id?: string;
  email: string;
  lastName: string;
  imageUrl?: string;
  userName?: string;
  firstName: string;
};

declare type Message = {
  id: number;
  content: string;
  type: "client" | "bot";
};
