export const isBrowser =
  typeof window !== "undefined" &&
  typeof window.localStorage !== "undefined" &&
  typeof window.sessionStorage !== "undefined";

export const isServer = !isBrowser;
