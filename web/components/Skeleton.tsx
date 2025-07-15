import React from "react";
import { cn } from "@/lib/utils";

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "text" | "circular" | "rectangular" | "rounded";
  width?: string | number;
  height?: string | number;
  animation?: "pulse" | "wave" | "none";
}

/**
 * Skeleton component for loading states
 * Provides visual feedback while content is loading
 */
export function Skeleton({
  className,
  variant = "text",
  width,
  height,
  animation = "pulse",
  ...props
}: SkeletonProps) {
  const baseClasses = "bg-gray-200";

  const animationClasses = {
    pulse: "animate-pulse",
    wave: "animate-shimmer",
    none: "",
  };

  const variantClasses = {
    text: "rounded-md h-4",
    circular: "rounded-full",
    rectangular: "rounded-none",
    rounded: "rounded-lg",
  };

  const style: React.CSSProperties = {
    width: width || (variant === "circular" ? 40 : "100%"),
    height: height || (variant === "circular" ? 40 : variant === "text" ? 16 : 60),
  };

  return (
    <div
      className={cn(baseClasses, animationClasses[animation], variantClasses[variant], className)}
      style={style}
      {...props}
    />
  );
}

/**
 * Skeleton container for grouping multiple skeletons
 */
export function SkeletonContainer({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("space-y-3", className)}>{children}</div>;
}

/**
 * Pre-built skeleton templates for common UI patterns
 */
export const SkeletonTemplates = {
  // Card skeleton
  Card: () => (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-4">
      <Skeleton variant="rectangular" height={200} className="mb-4" />
      <Skeleton variant="text" width="60%" />
      <Skeleton variant="text" width="80%" />
      <Skeleton variant="text" width="40%" />
    </div>
  ),

  // List item skeleton
  ListItem: () => (
    <div className="flex items-center space-x-4 p-4 bg-white rounded-lg">
      <Skeleton variant="circular" width={48} height={48} />
      <div className="flex-1 space-y-2">
        <Skeleton variant="text" width="40%" />
        <Skeleton variant="text" width="60%" />
      </div>
    </div>
  ),

  // Table row skeleton
  TableRow: () => (
    <tr className="border-b">
      <td className="p-4">
        <Skeleton variant="text" width="80%" />
      </td>
      <td className="p-4">
        <Skeleton variant="text" width="60%" />
      </td>
      <td className="p-4">
        <Skeleton variant="text" width="40%" />
      </td>
      <td className="p-4">
        <Skeleton variant="text" width="30%" />
      </td>
    </tr>
  ),

  // Article skeleton
  Article: () => (
    <article className="space-y-4">
      <Skeleton variant="text" height={32} width="70%" className="mb-6" />
      <Skeleton variant="rectangular" height={300} className="mb-6" />
      <SkeletonContainer>
        <Skeleton variant="text" />
        <Skeleton variant="text" />
        <Skeleton variant="text" width="90%" />
        <Skeleton variant="text" width="95%" />
        <Skeleton variant="text" width="85%" />
      </SkeletonContainer>
    </article>
  ),

  // User profile skeleton
  UserProfile: () => (
    <div className="flex items-start space-x-4">
      <Skeleton variant="circular" width={80} height={80} />
      <div className="flex-1 space-y-2">
        <Skeleton variant="text" width="30%" height={24} />
        <Skeleton variant="text" width="50%" />
        <Skeleton variant="text" width="40%" />
      </div>
    </div>
  ),

  // Dashboard stats skeleton
  DashboardStats: () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {[1, 2, 3].map((i) => (
        <div key={i} className="bg-white rounded-lg shadow p-6">
          <Skeleton variant="text" width="60%" className="mb-2" />
          <Skeleton variant="text" height={32} width="40%" className="mb-4" />
          <Skeleton variant="rectangular" height={100} />
        </div>
      ))}
    </div>
  ),
};

/**
 * Add shimmer animation to tailwind.config.js:
 *
 * animation: {
 *   shimmer: 'shimmer 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
 * },
 * keyframes: {
 *   shimmer: {
 *     '0%, 100%': { opacity: 1 },
 *     '50%': { opacity: 0.5 },
 *   },
 * },
 */
