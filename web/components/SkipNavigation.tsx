import React from "react";

/**
 * Skip Navigation component for keyboard accessibility
 * Allows users to skip repetitive navigation and jump to main content
 */
export function SkipNavigation() {
  return (
    <>
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
      >
        Skip to main content
      </a>
      <a
        href="#footer"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
      >
        Skip to footer
      </a>
    </>
  );
}

/**
 * Main landmark component
 */
export function MainContent({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <main
      id="main-content"
      tabIndex={-1}
      className={`focus:outline-none ${className}`}
      role="main"
      aria-label="Main content"
    >
      {children}
    </main>
  );
}

/**
 * Navigation landmark component
 */
export function Navigation({
  children,
  label,
  className = "",
}: {
  children: React.ReactNode;
  label: string;
  className?: string;
}) {
  return (
    <nav className={className} role="navigation" aria-label={label}>
      {children}
    </nav>
  );
}

/**
 * Footer landmark component
 */
export function Footer({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <footer id="footer" className={className} role="contentinfo" aria-label="Site footer">
      {children}
    </footer>
  );
}

/**
 * Header landmark component
 */
export function Header({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <header className={className} role="banner" aria-label="Site header">
      {children}
    </header>
  );
}

/**
 * Aside landmark component
 */
export function Aside({
  children,
  label,
  className = "",
}: {
  children: React.ReactNode;
  label: string;
  className?: string;
}) {
  return (
    <aside className={className} role="complementary" aria-label={label}>
      {children}
    </aside>
  );
}

/**
 * Search landmark component
 */
export function Search({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <search className={className} role="search" aria-label="Site search">
      {children}
    </search>
  );
}
