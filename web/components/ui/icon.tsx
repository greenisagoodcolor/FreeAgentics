/**
 * SSR-safe wrapper for Lucide React icons
 * Prevents hydration warnings by suppressing server-side styling conflicts
 */

import React from 'react';
import { LucideProps } from 'lucide-react';

interface IconProps extends LucideProps {
  icon: React.ComponentType<LucideProps>;
  suppressHydrationWarning?: boolean;
}

/**
 * Wrapper component for Lucide icons that handles SSR hydration properly
 * Prevents "Extra attributes from the server" warnings by suppressing hydration
 * for style-related attributes that may differ between server and client
 */
export function Icon({ 
  icon: IconComponent, 
  suppressHydrationWarning = true, 
  ...props 
}: IconProps) {
  return (
    <IconComponent 
      {...props} 
      suppressHydrationWarning={suppressHydrationWarning}
    />
  );
}

/**
 * Alternative approach: Create a client-only icon wrapper
 * This completely avoids SSR rendering for icons
 */
export function ClientIcon({ 
  icon: IconComponent, 
  ...props 
}: Omit<IconProps, 'suppressHydrationWarning'>) {
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    // Return placeholder during SSR
    return <div className={`inline-block ${props.className || 'w-4 h-4'}`} />;
  }

  return <IconComponent {...props} />;
}