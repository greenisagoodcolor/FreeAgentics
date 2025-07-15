# FreeAgentics Infrastructure Requirements

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [Network Configuration](#network-configuration)
3. [Load Balancer Setup](#load-balancer-setup)
4. [CDN Configuration](#cdn-configuration)
5. [Scaling Guidelines](#scaling-guidelines)
6. [High Availability Setup](#high-availability-setup)
7. [Monitoring Infrastructure](#monitoring-infrastructure)
8. [Backup and Disaster Recovery](#backup-and-disaster-recovery)

## Hardware Requirements

### Minimum Production Requirements

#### Application Servers
- **CPU**: 4 vCPUs (2.4 GHz or higher)
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps connection
- **Count**: Minimum 2 servers for redundancy

#### Database Server
- **CPU**: 8 vCPUs (2.4 GHz or higher)
- **Memory**: 32 GB RAM
- **Storage**: 500 GB SSD (NVMe preferred)
- **IOPS**: Minimum 3000 IOPS
- **Network**: 10 Gbps connection

#### Redis Cache Server
- **CPU**: 4 vCPUs
- **Memory**: 16 GB RAM (primarily for cache)
- **Storage**: 50 GB SSD
- **Network**: 10 Gbps connection

#### Load Balancer
- **CPU**: 2 vCPUs
- **Memory**: 4 GB RAM
- **Network**: 10 Gbps connection
- **Count**: 2 for HA configuration

### Recommended Production Setup

#### Application Cluster
```yaml
Application Servers (Auto-scaling group):
  - Instance Type: c5.xlarge (AWS) or equivalent
  - vCPUs: 4
  - Memory: 8 GB
  - Storage: 100 GB GP3 SSD
  - Min Instances: 3
  - Max Instances: 10
  - Target CPU: 70%
```

#### Database Cluster
```yaml
Primary Database:
  - Instance Type: r5.2xlarge (AWS) or equivalent
  - vCPUs: 8
  - Memory: 64 GB
  - Storage: 1 TB GP3 SSD (10,000 IOPS)
  
Read Replicas (2):
  - Instance Type: r5.xlarge
  - vCPUs: 4
  - Memory: 32 GB
  - Storage: 1 TB GP3 SSD
```

#### Cache Layer
```yaml
Redis Cluster:
  - Instance Type: r6g.xlarge (AWS ElastiCache)
  - Memory: 32 GB
  - Cluster Mode: Enabled
  - Shards: 3
  - Replicas per shard: 2
```

### Storage Requirements

#### Application Storage
- **OS and Application**: 50 GB
- **Logs**: 50 GB (with rotation)
- **Temporary Files**: 20 GB
- **Container Images**: 30 GB

#### Database Storage
- **Data**: 500 GB (with 50% growth headroom)
- **WAL/Logs**: 100 GB
- **Backups**: 1 TB (separate volume)
- **Growth Rate**: Plan for 20% yearly growth

#### Backup Storage
- **Full Backups**: 7 daily backups (7 TB)
- **Incremental**: 30 days (2 TB)
- **Archive**: 12 monthly backups (12 TB)
- **Total**: ~21 TB with compression

## Network Configuration

### Network Architecture

```
Internet
    |
    ├── CDN (CloudFlare/AWS CloudFront)
    |
    ├── DDoS Protection
    |
    ├── Load Balancer (Layer 7)
    |     |
    |     ├── Web Servers (DMZ)
    |     |
    |     └── API Servers (Private Subnet)
    |           |
    |           ├── Redis Cluster (Private Subnet)
    |           |
    |           └── Database Cluster (Private Subnet)
    |
    └── VPN Gateway (Management Access)
```

### Subnet Configuration

#### Public Subnets
```yaml
Public Subnet A (us-east-1a):
  - CIDR: 10.0.1.0/24
  - Purpose: Load Balancers, NAT Gateways
  
Public Subnet B (us-east-1b):
  - CIDR: 10.0.2.0/24
  - Purpose: Load Balancers, NAT Gateways
```

#### Private Subnets
```yaml
Private App Subnet A (us-east-1a):
  - CIDR: 10.0.10.0/24
  - Purpose: Application Servers
  
Private App Subnet B (us-east-1b):
  - CIDR: 10.0.11.0/24
  - Purpose: Application Servers
  
Private Data Subnet A (us-east-1a):
  - CIDR: 10.0.20.0/24
  - Purpose: Database Primary
  
Private Data Subnet B (us-east-1b):
  - CIDR: 10.0.21.0/24
  - Purpose: Database Replicas
```

### Security Groups

#### Load Balancer Security Group
```yaml
Inbound Rules:
  - Port 80 (HTTP) from 0.0.0.0/0
  - Port 443 (HTTPS) from 0.0.0.0/0
  
Outbound Rules:
  - Port 8000 to App Security Group
```

#### Application Security Group
```yaml
Inbound Rules:
  - Port 8000 from Load Balancer SG
  - Port 22 from Bastion SG
  
Outbound Rules:
  - Port 5432 to Database SG
  - Port 6379 to Redis SG
  - Port 443 to 0.0.0.0/0 (External APIs)
```

#### Database Security Group
```yaml
Inbound Rules:
  - Port 5432 from App SG
  - Port 22 from Bastion SG
  
Outbound Rules:
  - None (Restricted)
```

#### Redis Security Group
```yaml
Inbound Rules:
  - Port 6379 from App SG
  
Outbound Rules:
  - None (Restricted)
```

### Firewall Rules

```bash
# Application Server Firewall (iptables)
# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow SSH from bastion
iptables -A INPUT -p tcp -s 10.0.0.10/32 --dport 22 -j ACCEPT

# Allow app port from load balancer
iptables -A INPUT -p tcp -s 10.0.1.0/24 --dport 8000 -j ACCEPT
iptables -A INPUT -p tcp -s 10.0.2.0/24 --dport 8000 -j ACCEPT

# Drop all other inbound
iptables -A INPUT -j DROP
```

## Load Balancer Setup

### HAProxy Configuration

```haproxy
global
    log 127.0.0.1:514 local0
    chroot /var/lib/haproxy
    maxconn 4000
    tune.ssl.default-dh-param 2048
    
defaults
    mode http
    log global
    option httplog
    option dontlognull
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    
frontend https_front
    bind *:443 ssl crt /etc/ssl/certs/freeagentics.pem
    redirect scheme https if !{ ssl_fc }
    
    # Security headers
    http-response set-header Strict-Transport-Security "max-age=63072000"
    http-response set-header X-Frame-Options "DENY"
    http-response set-header X-Content-Type-Options "nosniff"
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 100 }
    
    # ACLs
    acl is_api path_beg /api /v1
    acl is_websocket hdr(Upgrade) -i WebSocket
    
    # Use backends
    use_backend websocket_back if is_websocket
    use_backend api_back if is_api
    default_backend web_back
    
backend api_back
    balance roundrobin
    option httpchk GET /health
    
    # Sticky sessions for stateful operations
    cookie SERVERID insert indirect nocache
    
    server api1 10.0.10.10:8000 check cookie api1
    server api2 10.0.10.11:8000 check cookie api2
    server api3 10.0.11.10:8000 check cookie api3
    
backend websocket_back
    balance source
    option http-server-close
    option forceclose
    
    server ws1 10.0.10.10:8000 check
    server ws2 10.0.10.11:8000 check
    server ws3 10.0.11.10:8000 check
    
backend web_back
    balance roundrobin
    option httpchk GET /
    
    server web1 10.0.10.20:3000 check
    server web2 10.0.10.21:3000 check
```

### Nginx Configuration (Alternative)

```nginx
upstream api_backend {
    least_conn;
    server 10.0.10.10:8000 max_fails=3 fail_timeout=30s;
    server 10.0.10.11:8000 max_fails=3 fail_timeout=30s;
    server 10.0.11.10:8000 max_fails=3 fail_timeout=30s;
    
    keepalive 32;
}

upstream websocket_backend {
    ip_hash;
    server 10.0.10.10:8000;
    server 10.0.10.11:8000;
    server 10.0.11.10:8000;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=20r/m;

server {
    listen 443 ssl http2;
    server_name api.freeagentics.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Auth endpoints (stricter rate limiting)
    location /api/v1/auth/ {
        limit_req zone=auth_limit burst=5 nodelay;
        
        proxy_pass http://api_backend;
        # ... (same proxy settings)
    }
    
    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # WebSocket timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://api_backend;
        access_log off;
    }
}
```

### AWS Application Load Balancer (ALB)

```yaml
# CloudFormation template excerpt
ApplicationLoadBalancer:
  Type: AWS::ElasticLoadBalancingV2::LoadBalancer
  Properties:
    Name: freeagentics-alb
    Type: application
    Scheme: internet-facing
    IpAddressType: ipv4
    Subnets:
      - !Ref PublicSubnetA
      - !Ref PublicSubnetB
    SecurityGroups:
      - !Ref ALBSecurityGroup
    Tags:
      - Key: Name
        Value: freeagentics-alb

TargetGroup:
  Type: AWS::ElasticLoadBalancingV2::TargetGroup
  Properties:
    Name: freeagentics-api-tg
    Port: 8000
    Protocol: HTTP
    VpcId: !Ref VPC
    HealthCheckEnabled: true
    HealthCheckPath: /health
    HealthCheckIntervalSeconds: 30
    HealthCheckTimeoutSeconds: 10
    HealthyThresholdCount: 3
    UnhealthyThresholdCount: 2
    TargetType: ip
    
HTTPSListener:
  Type: AWS::ElasticLoadBalancingV2::Listener
  Properties:
    LoadBalancerArn: !Ref ApplicationLoadBalancer
    Port: 443
    Protocol: HTTPS
    Certificates:
      - CertificateArn: !Ref SSLCertificate
    DefaultActions:
      - Type: forward
        TargetGroupArn: !Ref TargetGroup
```

## CDN Configuration

### CloudFlare Configuration

```yaml
# CloudFlare Page Rules and Settings
Page Rules:
  - URL: api.freeagentics.com/*
    Settings:
      - SSL: Full (Strict)
      - Cache Level: Bypass
      - Security Level: High
      
  - URL: app.freeagentics.com/*
    Settings:
      - SSL: Full (Strict)
      - Cache Level: Standard
      - Browser Cache TTL: 4 hours
      - Edge Cache TTL: 1 hour
      
  - URL: *.freeagentics.com/static/*
    Settings:
      - Cache Level: Cache Everything
      - Browser Cache TTL: 1 year
      - Edge Cache TTL: 1 month

Firewall Rules:
  - Name: Block Bad Bots
    Expression: (cf.client.bot) and not (cf.verified_bot)
    Action: Block
    
  - Name: Rate Limit API
    Expression: (http.request.uri.path contains "/api/")
    Action: Rate Limit (100 requests per minute)
    
  - Name: Challenge Suspicious
    Expression: (cf.threat_score gt 30)
    Action: Challenge

DDoS Protection:
  - Sensitivity: High
  - Action: Challenge

WAF Rules:
  - OWASP Core Rule Set: Enabled
  - Sensitivity: Medium
```

### AWS CloudFront Configuration

```yaml
CloudFrontDistribution:
  Type: AWS::CloudFront::Distribution
  Properties:
    DistributionConfig:
      Enabled: true
      Comment: FreeAgentics CDN
      
      Origins:
        - Id: APIOrigin
          DomainName: !GetAtt ApplicationLoadBalancer.DNSName
          CustomOriginConfig:
            HTTPPort: 80
            HTTPSPort: 443
            OriginProtocolPolicy: https-only
            
        - Id: S3Origin
          DomainName: !GetAtt StaticBucket.RegionalDomainName
          S3OriginConfig:
            OriginAccessIdentity: !Sub origin-access-identity/cloudfront/${CloudFrontOAI}
      
      DefaultCacheBehavior:
        TargetOriginId: APIOrigin
        ViewerProtocolPolicy: redirect-to-https
        AllowedMethods:
          - GET
          - HEAD
          - OPTIONS
          - PUT
          - POST
          - PATCH
          - DELETE
        CachedMethods:
          - GET
          - HEAD
          - OPTIONS
        Compress: true
        ForwardedValues:
          QueryString: true
          Headers:
            - Authorization
            - Origin
            - Referer
          Cookies:
            Forward: all
            
      CacheBehaviors:
        - PathPattern: /static/*
          TargetOriginId: S3Origin
          ViewerProtocolPolicy: https-only
          AllowedMethods:
            - GET
            - HEAD
          CachedMethods:
            - GET
            - HEAD
          Compress: true
          DefaultTTL: 86400
          MaxTTL: 31536000
          
        - PathPattern: /ws/*
          TargetOriginId: APIOrigin
          ViewerProtocolPolicy: https-only
          AllowedMethods:
            - GET
            - HEAD
            - OPTIONS
            - PUT
            - POST
            - PATCH
            - DELETE
          ForwardedValues:
            QueryString: true
            Headers:
              - Authorization
              - Sec-WebSocket-Key
              - Sec-WebSocket-Version
              - Sec-WebSocket-Protocol
              - Sec-WebSocket-Accept
          
      WebACLId: !Ref WebACL
      
      ViewerCertificate:
        AcmCertificateArn: !Ref SSLCertificate
        SslSupportMethod: sni-only
        MinimumProtocolVersion: TLSv1.2_2021
```

## Scaling Guidelines

### Horizontal Scaling Strategy

#### Application Tier Scaling

```yaml
Auto Scaling Configuration:
  Metrics:
    - CPU Utilization > 70% for 5 minutes → Add 2 instances
    - CPU Utilization < 30% for 10 minutes → Remove 1 instance
    - Request count > 1000 req/s → Add 3 instances
    - Response time > 500ms for 5 minutes → Add 2 instances
    
  Scaling Policy:
    - Min Instances: 3
    - Max Instances: 20
    - Desired Capacity: 5
    - Cooldown Period: 300 seconds
    - Health Check Grace Period: 600 seconds
```

#### Database Scaling

```yaml
Read Replica Scaling:
  Triggers:
    - Read Query Load > 80% → Add read replica
    - Connection count > 80% of max → Add read replica
    
  Limits:
    - Max Read Replicas: 5
    - Replication Lag Threshold: 1 second
    
Vertical Scaling Thresholds:
  - CPU > 80% sustained → Upgrade instance class
  - Memory > 90% → Upgrade instance class
  - IOPS > 80% of provisioned → Increase IOPS
```

#### Redis Scaling

```yaml
Redis Cluster Scaling:
  Shard Addition Triggers:
    - Memory usage > 75% → Add shard
    - Network throughput > 80% → Add shard
    
  Node Addition Triggers:
    - CPU > 70% → Add replica nodes
    - Connection count > 10000 → Add replica nodes
```

### Vertical Scaling Guidelines

```yaml
Instance Upgrade Path:
  Application Servers:
    - t3.medium → t3.large → t3.xlarge → c5.xlarge → c5.2xlarge
    
  Database Servers:
    - db.t3.large → db.r5.xlarge → db.r5.2xlarge → db.r5.4xlarge
    
  Cache Servers:
    - cache.t3.medium → cache.r6g.large → cache.r6g.xlarge → cache.r6g.2xlarge
```

### Performance Benchmarks

```yaml
Target Metrics:
  API Response Time:
    - p50: < 100ms
    - p95: < 300ms
    - p99: < 500ms
    
  Throughput:
    - Minimum: 1000 req/s
    - Target: 5000 req/s
    - Peak: 10000 req/s
    
  Concurrent Users:
    - Normal: 1000
    - Peak: 5000
    - Maximum: 10000
```

## High Availability Setup

### Multi-Region Architecture

```yaml
Primary Region (us-east-1):
  - Full application stack
  - Primary database
  - Primary Redis cluster
  - Active traffic serving
  
Secondary Region (us-west-2):
  - Full application stack
  - Read replica database
  - Standby Redis cluster
  - Ready for failover
  
Database Replication:
  - Type: Asynchronous
  - Lag Target: < 1 second
  - Failover RTO: < 5 minutes
  - Failover RPO: < 1 minute
```

### Failover Procedures

```bash
# Automated failover script
#!/bin/bash
# scripts/deployment/failover-to-secondary.sh

# 1. Promote secondary database
aws rds promote-read-replica \
  --db-instance-identifier freeagentics-west-replica

# 2. Update DNS to point to secondary region
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch file://failover-dns-changes.json

# 3. Warm up secondary caches
./scripts/cache-warmer.sh us-west-2

# 4. Verify secondary health
./scripts/health-check.sh us-west-2
```

## Monitoring Infrastructure

### Prometheus Setup

```yaml
Prometheus Servers:
  Primary:
    - Instance: t3.large
    - Storage: 500 GB
    - Retention: 30 days
    
  Secondary:
    - Instance: t3.medium
    - Storage: 200 GB
    - Retention: 7 days
    
Scrape Configs:
  - job_name: 'api-servers'
    scrape_interval: 15s
    static_configs:
      - targets: ['api1:9090', 'api2:9090', 'api3:9090']
      
  - job_name: 'database'
    scrape_interval: 30s
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    scrape_interval: 15s
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

```yaml
Required Dashboards:
  - System Overview (CPU, Memory, Disk, Network)
  - API Performance (Response times, Error rates)
  - Database Performance (Query times, Connections)
  - Redis Performance (Hit rate, Memory usage)
  - Business Metrics (Active users, Transactions)
  - Security Dashboard (Failed auth, Suspicious activity)
```

### Alerting Rules

```yaml
Critical Alerts:
  - API Down: Immediate page
  - Database Down: Immediate page
  - Disk > 90%: Immediate page
  - Security Breach: Immediate page
  
Warning Alerts:
  - CPU > 80%: Email + Slack
  - Memory > 85%: Email + Slack
  - Response Time > 1s: Email
  - Error Rate > 5%: Email + Slack
```

## Backup and Disaster Recovery

### Backup Strategy

```yaml
Database Backups:
  Full Backup:
    - Frequency: Daily at 2 AM UTC
    - Retention: 7 days local, 30 days S3
    - Type: pg_dump with compression
    
  Incremental Backup:
    - Frequency: Every 6 hours
    - Retention: 48 hours
    - Type: WAL archiving
    
  Point-in-Time Recovery:
    - Available for: Last 7 days
    - RPO: 5 minutes
    
Application Data:
  Code Repositories:
    - Mirror to secondary Git server
    - Backup to S3 daily
    
  User Uploads:
    - Real-time sync to S3
    - Cross-region replication enabled
    
  Configuration:
    - Stored in version control
    - Encrypted backups to S3
```

### Disaster Recovery Plan

```yaml
RTO/RPO Targets:
  - Recovery Time Objective (RTO): 1 hour
  - Recovery Point Objective (RPO): 15 minutes
  
DR Scenarios:
  1. Single Server Failure:
     - Detection: < 1 minute
     - Failover: Automatic via load balancer
     - Data Loss: None
     
  2. Database Failure:
     - Detection: < 1 minute
     - Failover: 5-10 minutes to promote replica
     - Data Loss: < 1 minute
     
  3. Region Failure:
     - Detection: < 5 minutes
     - Failover: 15-30 minutes to secondary region
     - Data Loss: < 5 minutes
     
  4. Complete Infrastructure Loss:
     - Recovery: 2-4 hours from backups
     - Data Loss: < 1 hour
```

### Recovery Procedures

```bash
# Database recovery from backup
#!/bin/bash
# scripts/dr/recover-database.sh

# 1. Restore latest full backup
aws s3 cp s3://backups/db/latest-full.sql.gz .
gunzip latest-full.sql.gz
psql -h localhost -U postgres -d postgres -c "CREATE DATABASE freeagentics_restore"
psql -h localhost -U postgres -d freeagentics_restore < latest-full.sql

# 2. Apply incremental changes
aws s3 sync s3://backups/db/wal/ /var/lib/postgresql/wal/
pg_wal_replay

# 3. Verify data integrity
./scripts/verify-database-integrity.sh

# 4. Switch application to restored database
./scripts/switch-database.sh freeagentics_restore
```

## Cost Optimization

### Reserved Instances
- Commit to 1-year or 3-year terms for predictable workloads
- Expected savings: 30-50%

### Spot Instances
- Use for non-critical batch processing
- Worker nodes that can handle interruptions
- Expected savings: 70-90%

### Auto-scaling Optimization
- Scale down during off-peak hours
- Use predictive scaling for known patterns
- Implement proper cooldown periods

### Storage Optimization
- Use lifecycle policies for logs and backups
- Compress old data
- Move infrequent data to cheaper storage tiers

### Monitoring Costs
```yaml
Cost Alerts:
  - Daily spend > $500: Email finance team
  - Monthly projection > $15,000: Review meeting
  - Unusual spike (>50% increase): Immediate investigation
```