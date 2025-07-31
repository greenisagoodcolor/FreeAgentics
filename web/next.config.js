// Safe import for optional bundle analyzer (dev mode support)
let withBundleAnalyzer;
try {
  withBundleAnalyzer = require("@next/bundle-analyzer")({
    enabled: process.env.ANALYZE === "true",
  });
} catch (e) {
  // Bundle analyzer is optional - skip if not installed
  withBundleAnalyzer = (config) => config;
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone", // Enable standalone output for Docker deployments

  // Skip build-time prerendering of API routes and sitemap for CI
  skipTrailingSlashRedirect: true,
  trailingSlash: false,

  // Compiler optimizations
  compiler: {
    removeConsole: process.env.NODE_ENV === "production",
  },

  // Bundle optimization
  webpack: (config, { isServer }) => {
    // Bundle analyzer configuration
    if (process.env.ANALYZE === "true") {
      config.optimization.splitChunks = {
        chunks: "all",
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: "vendors",
            chunks: "all",
          },
        },
      };
    }

    // Optimization for production (simplified per CLAUDE.md KISS principle)
    if (process.env.NODE_ENV === "production") {
      config.optimization = {
        ...config.optimization,
        minimize: true,
        // Standard chunk splitting
        splitChunks: {
          chunks: "all",
          minSize: 10000,
          maxSize: 50000,
        },
      };
    }

    // Reduce bundle size with specific optimizations
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        crypto: false,
        stream: false,
        util: false,
        buffer: false,
        process: false,
      };

      // Use production builds
      if (process.env.NODE_ENV === "production") {
        config.resolve.alias = {
          ...config.resolve.alias,
        };
      }

      // Module concatenation for better compression
      config.optimization.concatenateModules = true;
    }

    return config;
  },

  // API proxy to backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.BACKEND_URL || "http://localhost:8000/api/:path*",
      },
      // WebSocket proxy removed - handled by client-side connection
    ];
  },

  // CORS headers - SECURITY HARDENED: Restricted to specific origins
  async headers() {
    const isDev = process.env.NODE_ENV !== "production";

    return [
      {
        source: "/api/:path*",
        headers: [
          // Only require credentials in production
          ...(isDev ? [] : [{ key: "Access-Control-Allow-Credentials", value: "true" }]),
          {
            key: "Access-Control-Allow-Origin",
            value: isDev ? "*" : process.env.ALLOWED_ORIGINS || "https://yourdomain.com",
          },
          {
            key: "Access-Control-Allow-Methods",
            value: "GET,OPTIONS,PATCH,DELETE,POST,PUT",
          },
          {
            key: "Access-Control-Allow-Headers",
            value:
              "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version",
          },
        ],
      },
    ];
  },
};

module.exports = withBundleAnalyzer(nextConfig);
