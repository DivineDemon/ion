import { useEffect } from "react";

import { useLocalStorage } from "usehooks-ts";

import { api } from "@/trpc/react";

const useAccount = () => {
  const [account, setAccount] = useLocalStorage("account", "");
  const { data: accounts } = api.user.fetchUserLinkedAccounts.useQuery();

  useEffect(() => {
    if (accounts?.grantEmail?.length && !account) {
      setAccount(accounts.grantEmail[0] as string);
    }
  }, [accounts, account, setAccount]);

  return {
    account,
    accounts,
    setAccount,
  };
};

export default useAccount;
