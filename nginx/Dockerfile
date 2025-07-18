FROM nginx:1.25-alpine

# Install dependencies
RUN apk add --no-cache \
    curl \
    openssl \
    certbot \
    certbot-nginx

# Remove default nginx config
RUN rm -f /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/*.conf /etc/nginx/conf.d/
COPY snippets/*.conf /etc/nginx/snippets/

# Copy SSL certificates (in production, these should be mounted)
COPY ssl/dhparam.pem /etc/nginx/dhparam.pem

# Create directories for Let's Encrypt
RUN mkdir -p /var/www/certbot /etc/letsencrypt

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Create non-root user
RUN adduser -D -H -u 1000 -s /sbin/nologin nginx-user

# Set correct permissions
RUN chown -R nginx-user:nginx-user /var/cache/nginx /var/log/nginx /etc/nginx/conf.d && \
    chmod -R 755 /var/cache/nginx && \
    touch /var/run/nginx.pid && \
    chown nginx-user:nginx-user /var/run/nginx.pid

# Switch to non-root user
USER nginx-user

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
