import * as React from "react";

const MOBILE_BREAKPOINT = 768;

export function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState<boolean | undefined>(
    undefined,
  );

  React.useEffect(() => {
    // Check if matchMedia is available (for SSR and testing environments)
    if (typeof window !== "undefined" && window.matchMedia) {
      const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`);
      const handleChange = () => {
        setIsMobile(mql.matches);
      };
      mql.addEventListener("change", handleChange);
      setIsMobile(mql.matches);
      return () => mql.removeEventListener("change", handleChange);
    } else {
      // Fallback for environments without matchMedia
      setIsMobile(false);
    }
  }, []);

  return !!isMobile;
}
