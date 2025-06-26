"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Home, Settings } from "lucide-react";

export default function NavBar() {
  const pathname = usePathname();

  return (
    <div className="fixed top-0 right-0 p-4 z-50 flex gap-2">
      <Button
        variant={pathname === "/" ? "default" : "outline"}
        size="sm"
        asChild
      >
        <Link href="/">
          <Home size={16} className="mr-2" />
          Home
        </Link>
      </Button>

      <Button
        variant={pathname === "/settings" ? "default" : "outline"}
        size="sm"
        asChild
      >
        <Link href="/settings">
          <Settings size={16} className="mr-2" />
          Settings
        </Link>
      </Button>
    </div>
  );
}
