"use client";

import useDevAuth from "@/hooks/use-dev-auth";

export default function DevAuthBootstrap() {
  useDevAuth();
  return null;
}