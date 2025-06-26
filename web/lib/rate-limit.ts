/**
 * Simple in-memory rate limiter
 * For production, use Redis or similar
 */

interface RateLimitOptions {
  interval: number; // Time window in milliseconds
  uniqueTokenPerInterval: number; // Max number of tokens
}

interface RateLimitStore {
  [key: string]: {
    count: number;
    resetTime: number;
  };
}

class RateLimiter {
  private store: RateLimitStore = {};
  private interval: number;
  private limit: number;

  constructor(options: RateLimitOptions) {
    this.interval = options.interval;
    this.limit = options.uniqueTokenPerInterval;
  }

  async check(tokens: number, identifier: string): Promise<void> {
    const now = Date.now();
    const record = this.store[identifier];

    if (!record || now > record.resetTime) {
      // Create new record or reset expired one
      this.store[identifier] = {
        count: tokens,
        resetTime: now + this.interval,
      };
      return;
    }

    if (record.count + tokens > this.limit) {
      const waitTime = record.resetTime - now;
      throw new Error(`Rate limit exceeded. Try again in ${Math.ceil(waitTime / 1000)} seconds.`);
    }

    record.count += tokens;
  }

  getRemainingTokens(identifier: string): number {
    const now = Date.now();
    const record = this.store[identifier];

    if (!record || now > record.resetTime) {
      return this.limit;
    }

    return Math.max(0, this.limit - record.count);
  }

  getResetTime(identifier: string): number {
    const record = this.store[identifier];
    return record ? record.resetTime : Date.now() + this.interval;
  }

  // Clean up expired entries periodically
  cleanup(): void {
    const now = Date.now();
    for (const key in this.store) {
      if (this.store[key] && this.store[key].resetTime < now) {
        delete this.store[key];
      }
    }
  }
}

// Factory function
export function rateLimit(options: RateLimitOptions): RateLimiter {
  const limiter = new RateLimiter(options);

  // Run cleanup every minute
  setInterval(() => limiter.cleanup(), 60000);

  return limiter;
}

export type { RateLimiter, RateLimitOptions };
