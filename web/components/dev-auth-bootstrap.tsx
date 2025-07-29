"use client";

import useDevAuth from "@/lib/useDevAuth";

export default function DevAuthBootstrap() {
  useDevAuth();
  return null;
}