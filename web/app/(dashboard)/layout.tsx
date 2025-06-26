import { Metadata } from 'next';
import { Suspense } from 'react';

export const metadata: Metadata = {
  title: 'Dashboard | CogniticNet',
  description: 'Real-time multi-agent simulation dashboard',
};

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="dashboard-layout">
      <Suspense fallback={
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="animate-pulse">Loading dashboard...</div>
        </div>
      }>
        {children}
      </Suspense>
    </div>
  );
}
