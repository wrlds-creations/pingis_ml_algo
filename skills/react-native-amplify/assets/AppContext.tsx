import React, { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import { fetchUserAttributes, getCurrentUser, signOut } from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';

type AppUser = {
  id: string;
  email: string;
  name?: string;
  role?: string;
};

type AppContextValue = {
  user: AppUser | null;
  isLoading: boolean;
  refreshUser: () => Promise<void>;
  signOutUser: () => Promise<void>;
};

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AppUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  async function refreshUser() {
    setIsLoading(true);
    try {
      const authUser = await getCurrentUser();
      const attributes = await fetchUserAttributes();

      setUser({
        id: authUser.userId,
        email: attributes.email ?? '',
        name: attributes.name,
        role: attributes.profile,
      });
    } catch {
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }

  async function signOutUser() {
    await signOut();
  }

  useEffect(() => {
    void refreshUser();

    const unsubscribe = Hub.listen('auth', ({ payload }) => {
      if (payload.event === 'signedIn') void refreshUser();
      if (payload.event === 'signedOut') setUser(null);
    });

    return () => unsubscribe();
  }, []);

  return (
    <AppContext.Provider value={{ user, isLoading, refreshUser, signOutUser }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
}
