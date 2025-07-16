/**
 * Service Worker for FreeAgentics
 * Provides offline support and caching for improved performance
 */

const CACHE_NAME = "freeagentics-v1";
const STATIC_CACHE_NAME = "freeagentics-static-v1";

// Define what to cache
const STATIC_ASSETS = [
  "/",
  "/dashboard",
  "/agents",
  "/manifest.json",
  "/icons/icon-192x192.png",
  "/icons/icon-512x512.png",
];

// API routes that should be cached
const API_CACHE_PATTERNS = [/^\/api\/health/, /^\/api\/agents\/\w+$/];

// Network-first strategy for these patterns
const NETWORK_FIRST_PATTERNS = [/^\/api\/agents$/, /^\/api\/conversations/, /^\/api\/knowledge/];

// Cache-first strategy for these patterns
const CACHE_FIRST_PATTERNS = [
  /\.(js|css|woff|woff2|eot|ttf|otf)$/,
  /\.(png|jpg|jpeg|gif|svg|ico|webp|avif)$/,
  /^\/_next\/static\//,
  /^\/static\//,
];

// Install event - cache static assets
self.addEventListener("install", (event) => {
  console.log("Service Worker installing...");

  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE_NAME).then((cache) => {
        return cache.addAll(STATIC_ASSETS);
      }),
      caches.open(CACHE_NAME).then((cache) => {
        // Pre-cache critical API endpoints
        return Promise.all([cache.add("/api/health")]);
      }),
    ]),
  );

  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  console.log("Service Worker activating...");

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME && cacheName !== STATIC_CACHE_NAME) {
            console.log("Deleting old cache:", cacheName);
            return caches.delete(cacheName);
          }
        }),
      );
    }),
  );

  self.clients.claim();
});

// Fetch event - handle requests with appropriate caching strategy
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // Handle different request types
  if (request.method === "GET") {
    event.respondWith(handleGetRequest(request));
  } else if (request.method === "POST" || request.method === "PUT" || request.method === "DELETE") {
    event.respondWith(handleMutationRequest(request));
  }
});

/**
 * Handle GET requests with appropriate caching strategy
 */
async function handleGetRequest(request) {
  const url = new URL(request.url);
  const pathname = url.pathname;

  try {
    // Cache-first strategy for static assets
    if (CACHE_FIRST_PATTERNS.some((pattern) => pattern.test(pathname))) {
      return await cacheFirst(request);
    }

    // Network-first strategy for dynamic API data
    if (NETWORK_FIRST_PATTERNS.some((pattern) => pattern.test(pathname))) {
      return await networkFirst(request);
    }

    // Stale-while-revalidate for API endpoints that can be cached
    if (API_CACHE_PATTERNS.some((pattern) => pattern.test(pathname))) {
      return await staleWhileRevalidate(request);
    }

    // Network-only for all other requests
    return await networkOnly(request);
  } catch (error) {
    console.error("Error handling request:", error);
    return await handleOfflineRequest(request);
  }
}

/**
 * Handle POST/PUT/DELETE requests with network-first approach
 */
async function handleMutationRequest(request) {
  try {
    const response = await fetch(request);

    // If successful, invalidate related cache entries
    if (response.ok) {
      await invalidateRelatedCache(request);
    }

    return response;
  } catch (error) {
    console.error("Mutation request failed:", error);
    return new Response(JSON.stringify({ error: "Network error" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}

/**
 * Cache-first strategy
 */
async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  if (cached) {
    return cached;
  }

  const response = await fetch(request);
  if (response.ok) {
    cache.put(request, response.clone());
  }

  return response;
}

/**
 * Network-first strategy
 */
async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME);

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    throw error;
  }
}

/**
 * Stale-while-revalidate strategy
 */
async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  });

  return cached || fetchPromise;
}

/**
 * Network-only strategy
 */
async function networkOnly(request) {
  return fetch(request);
}

/**
 * Handle offline requests
 */
async function handleOfflineRequest(request) {
  const url = new URL(request.url);

  // Serve cached version if available
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  if (cached) {
    return cached;
  }

  // Return offline page for navigation requests
  if (request.mode === "navigate") {
    const offlineResponse = await cache.match("/");
    if (offlineResponse) {
      return offlineResponse;
    }
  }

  // Return generic offline response
  return new Response(
    JSON.stringify({
      error: "Offline",
      message: "This content is not available offline",
    }),
    {
      status: 503,
      headers: { "Content-Type": "application/json" },
    },
  );
}

/**
 * Invalidate related cache entries after mutations
 */
async function invalidateRelatedCache(request) {
  const url = new URL(request.url);
  const cache = await caches.open(CACHE_NAME);

  // Invalidate related endpoints
  if (url.pathname.includes("/agents")) {
    await cache.delete("/api/agents");
  }

  if (url.pathname.includes("/conversations")) {
    await cache.delete("/api/conversations");
  }

  if (url.pathname.includes("/knowledge")) {
    await cache.delete("/api/knowledge");
  }
}

// Background sync for failed requests
self.addEventListener("sync", (event) => {
  if (event.tag === "background-sync") {
    event.waitUntil(syncFailedRequests());
  }
});

/**
 * Handle background sync of failed requests
 */
async function syncFailedRequests() {
  // This would typically sync queued requests from IndexedDB
  console.log("Background sync triggered");
}

// Push notifications
self.addEventListener("push", (event) => {
  if (event.data) {
    const data = event.data.json();

    event.waitUntil(
      self.registration.showNotification(data.title, {
        body: data.body,
        icon: "/icons/icon-192x192.png",
        badge: "/icons/icon-72x72.png",
        tag: data.tag || "default",
        requireInteraction: data.requireInteraction || false,
        actions: data.actions || [],
      }),
    );
  }
});

// Notification click handler
self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  if (event.action === "view") {
    event.waitUntil(clients.openWindow(event.notification.data.url || "/"));
  }
});

// Message handler for communication with main thread
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

// Performance monitoring
self.addEventListener("fetch", (event) => {
  const start = Date.now();

  event.respondWith(
    handleRequest(event.request).then((response) => {
      const duration = Date.now() - start;

      // Log performance metrics
      if (duration > 1000) {
        console.warn(`Slow request: ${event.request.url} took ${duration}ms`);
      }

      return response;
    }),
  );
});

/**
 * Main request handler
 */
async function handleRequest(request) {
  // Use the appropriate strategy based on request type
  if (request.method === "GET") {
    return await handleGetRequest(request);
  } else {
    return await handleMutationRequest(request);
  }
}
