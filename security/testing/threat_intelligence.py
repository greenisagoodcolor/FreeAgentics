"""
Threat Intelligence Integration

Integrates with threat intelligence feeds to identify and block malicious actors.
Includes IP reputation checking, real-time blocking, and threat correlation.
"""

import asyncio
import ipaddress
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
import redis.asyncio as redis
from bloom_filter2 import BloomFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1

    @property
    def name(self) -> str:
        return super().name.lower()


class ThreatType(Enum):
    """Types of threats"""

    MALWARE = "malware"
    BOTNET = "botnet"
    SPAM = "spam"
    SCANNER = "scanner"
    BRUTE_FORCE = "brute_force"
    EXPLOIT = "exploit"
    PROXY = "proxy"
    TOR = "tor"
    VPN = "vpn"
    COMPROMISED = "compromised"
    REPUTATION = "reputation"


@dataclass
class ThreatIndicator:
    """Threat indicator information"""

    indicator: str
    indicator_type: str  # ip, domain, url, hash
    threat_types: List[ThreatType]
    threat_level: ThreatLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expire_time: Optional[datetime] = None


@dataclass
class ThreatIntelConfig:
    """Configuration for threat intelligence"""

    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    block_duration: int = 86400  # 24 hours
    auto_block_threshold: ThreatLevel = ThreatLevel.HIGH
    enable_otx: bool = True
    otx_api_key: Optional[str] = None
    enable_abuseipdb: bool = True
    abuseipdb_api_key: Optional[str] = None
    enable_virustotal: bool = False
    virustotal_api_key: Optional[str] = None
    custom_feeds: List[str] = field(default_factory=list)
    whitelist_ips: Set[str] = field(default_factory=set)
    whitelist_domains: Set[str] = field(default_factory=set)
    max_requests_per_minute: int = 60
    bloom_filter_size: int = 10000000  # 10M items
    bloom_filter_fp_rate: float = 0.01


class ThreatFeedManager:
    """Manages threat intelligence feeds"""

    def __init__(self, config: ThreatIntelConfig):
        self.config = config
        self.feeds = []

        # Initialize feeds
        if config.enable_otx and config.otx_api_key:
            self.feeds.append(OTXFeed(config))

        if config.enable_abuseipdb and config.abuseipdb_api_key:
            self.feeds.append(AbuseIPDBFeed(config))

        if config.enable_virustotal and config.virustotal_api_key:
            self.feeds.append(VirusTotalFeed(config))

        # Add custom feeds
        for feed_url in config.custom_feeds:
            self.feeds.append(CustomFeed(feed_url, config))

    async def fetch_all_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from all feeds"""
        all_indicators = []

        # Fetch from all feeds concurrently
        tasks = [feed.fetch_indicators() for feed in self.feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error fetching from feed {self.feeds[i].__class__.__name__}: {result}"
                )
            else:
                all_indicators.extend(result)

        # Deduplicate indicators
        unique_indicators = self._deduplicate_indicators(all_indicators)

        logger.info(
            f"Fetched {len(unique_indicators)} unique indicators from {len(self.feeds)} feeds"
        )
        return unique_indicators

    def _deduplicate_indicators(
        self, indicators: List[ThreatIndicator]
    ) -> List[ThreatIndicator]:
        """Deduplicate indicators, keeping highest threat level"""
        indicator_map = {}

        for indicator in indicators:
            key = (indicator.indicator, indicator.indicator_type)

            if key in indicator_map:
                # Keep the one with higher threat level
                existing = indicator_map[key]
                if indicator.threat_level.value > existing.threat_level.value:
                    indicator_map[key] = indicator
                else:
                    # Merge threat types and sources
                    existing.threat_types = list(
                        set(existing.threat_types + indicator.threat_types)
                    )
                    existing.tags = list(set(existing.tags + indicator.tags))
                    existing.last_seen = max(
                        existing.last_seen, indicator.last_seen
                    )
                    existing.confidence = max(
                        existing.confidence, indicator.confidence
                    )
            else:
                indicator_map[key] = indicator

        return list(indicator_map.values())


class OTXFeed:
    """AlienVault OTX threat feed"""

    def __init__(self, config: ThreatIntelConfig):
        self.config = config
        self.api_url = "https://otx.alienvault.com/api/v1"

    async def fetch_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from OTX"""
        indicators = []

        try:
            headers = {"X-OTX-API-KEY": self.config.otx_api_key}

            async with aiohttp.ClientSession() as session:
                # Fetch subscribed pulses
                async with session.get(
                    f"{self.api_url}/pulses/subscribed",
                    headers=headers,
                    params={"limit": 100},
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        pulses = data.get("results", [])

                        for pulse in pulses:
                            pulse_indicators = self._parse_pulse(pulse)
                            indicators.extend(pulse_indicators)

        except Exception as e:
            logger.error(f"Error fetching OTX indicators: {e}")

        return indicators

    def _parse_pulse(self, pulse: Dict[str, Any]) -> List[ThreatIndicator]:
        """Parse OTX pulse into indicators"""
        indicators = []
        pulse_created = datetime.fromisoformat(
            pulse.get("created", "").replace("Z", "+00:00")
        )
        pulse_modified = datetime.fromisoformat(
            pulse.get("modified", "").replace("Z", "+00:00")
        )

        for indicator_data in pulse.get("indicators", []):
            indicator_type = indicator_data.get("type", "").lower()
            indicator_value = indicator_data.get("indicator", "")

            if not indicator_value:
                continue

            # Map OTX types to our types
            if indicator_type in ["ipv4", "ipv6"]:
                ind_type = "ip"
            elif indicator_type in ["domain", "hostname"]:
                ind_type = "domain"
            elif indicator_type == "url":
                ind_type = "url"
            elif indicator_type in [
                "filehash-md5",
                "filehash-sha1",
                "filehash-sha256",
            ]:
                ind_type = "hash"
            else:
                continue

            # Determine threat types from tags and description
            threat_types = self._extract_threat_types(
                pulse.get("tags", []) + [pulse.get("name", "")]
            )

            indicator = ThreatIndicator(
                indicator=indicator_value,
                indicator_type=ind_type,
                threat_types=threat_types,
                threat_level=self._determine_threat_level(pulse),
                source="OTX",
                first_seen=pulse_created,
                last_seen=pulse_modified,
                confidence=0.7,
                tags=pulse.get("tags", []),
                metadata={
                    "pulse_id": pulse.get("id"),
                    "pulse_name": pulse.get("name"),
                    "tlp": pulse.get("tlp", "white"),
                },
            )

            indicators.append(indicator)

        return indicators

    def _extract_threat_types(self, tags: List[str]) -> List[ThreatType]:
        """Extract threat types from tags"""
        threat_types = []
        tag_text = " ".join(tags).lower()

        threat_mapping = {
            ThreatType.MALWARE: ["malware", "trojan", "virus", "ransomware"],
            ThreatType.BOTNET: ["botnet", "c2", "c&c", "command"],
            ThreatType.SPAM: ["spam", "phishing", "scam"],
            ThreatType.SCANNER: ["scanner", "scanning", "recon"],
            ThreatType.EXPLOIT: ["exploit", "vulnerability", "cve"],
        }

        for threat_type, keywords in threat_mapping.items():
            if any(keyword in tag_text for keyword in keywords):
                threat_types.append(threat_type)

        if not threat_types:
            threat_types.append(ThreatType.REPUTATION)

        return threat_types

    def _determine_threat_level(self, pulse: Dict[str, Any]) -> ThreatLevel:
        """Determine threat level from pulse data"""
        # OTX doesn't provide explicit threat levels, so we estimate
        tags = " ".join(pulse.get("tags", [])).lower()

        if any(word in tags for word in ["critical", "apt", "targeted"]):
            return ThreatLevel.CRITICAL
        elif any(word in tags for word in ["high", "malware", "ransomware"]):
            return ThreatLevel.HIGH
        elif any(word in tags for word in ["medium", "suspicious"]):
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


class AbuseIPDBFeed:
    """AbuseIPDB threat feed"""

    def __init__(self, config: ThreatIntelConfig):
        self.config = config
        self.api_url = "https://api.abuseipdb.com/api/v2"

    async def fetch_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from AbuseIPDB"""
        indicators = []

        try:
            headers = {
                "Key": self.config.abuseipdb_api_key,
                "Accept": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                # Fetch blacklist
                params = {"confidenceMinimum": 75, "limit": 10000}

                async with session.get(
                    f"{self.api_url}/blacklist", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        for entry in data.get("data", []):
                            indicator = self._parse_entry(entry)
                            if indicator:
                                indicators.append(indicator)

        except Exception as e:
            logger.error(f"Error fetching AbuseIPDB indicators: {e}")

        return indicators

    def _parse_entry(self, entry: Dict[str, Any]) -> Optional[ThreatIndicator]:
        """Parse AbuseIPDB entry"""
        ip_address = entry.get("ipAddress")
        if not ip_address:
            return None

        # Map abuse categories to threat types
        categories = entry.get("abuseCategories", [])
        threat_types = self._map_abuse_categories(categories)

        return ThreatIndicator(
            indicator=ip_address,
            indicator_type="ip",
            threat_types=threat_types,
            threat_level=self._calculate_threat_level(entry),
            source="AbuseIPDB",
            first_seen=datetime.now()
            - timedelta(days=entry.get("numReports", 1)),
            last_seen=datetime.now(),
            confidence=entry.get("abuseConfidenceScore", 0) / 100.0,
            metadata={
                "country_code": entry.get("countryCode"),
                "usage_type": entry.get("usageType"),
                "isp": entry.get("isp"),
                "total_reports": entry.get("numReports"),
            },
        )

    def _map_abuse_categories(self, categories: List[int]) -> List[ThreatType]:
        """Map AbuseIPDB categories to threat types"""
        # AbuseIPDB category mappings
        category_mapping = {
            3: ThreatType.SPAM,  # Fraud Orders
            4: ThreatType.SCANNER,  # DDoS Attack
            5: ThreatType.SCANNER,  # FTP Brute-Force
            6: ThreatType.SCANNER,  # Ping of Death
            7: ThreatType.SCANNER,  # Phishing
            8: ThreatType.SPAM,  # Fraud VoIP
            9: ThreatType.PROXY,  # Open Proxy
            10: ThreatType.SPAM,  # Web Spam
            11: ThreatType.SPAM,  # Email Spam
            12: ThreatType.SPAM,  # Blog Spam
            13: ThreatType.PROXY,  # VPN IP
            14: ThreatType.SCANNER,  # Port Scan
            15: ThreatType.MALWARE,  # Hacking
            16: ThreatType.BRUTE_FORCE,  # SQL Injection
            17: ThreatType.SCANNER,  # Spoofing
            18: ThreatType.BRUTE_FORCE,  # Brute-Force
            19: ThreatType.SCANNER,  # Bad Web Bot
            20: ThreatType.EXPLOIT,  # Exploited Host
            21: ThreatType.SCANNER,  # Web App Attack
            22: ThreatType.BRUTE_FORCE,  # SSH
            23: ThreatType.BOTNET,  # IoT Targeted
        }

        threat_types = []
        for cat in categories:
            threat_type = category_mapping.get(cat)
            if threat_type and threat_type not in threat_types:
                threat_types.append(threat_type)

        if not threat_types:
            threat_types.append(ThreatType.REPUTATION)

        return threat_types

    def _calculate_threat_level(self, entry: Dict[str, Any]) -> ThreatLevel:
        """Calculate threat level from abuse score"""
        score = entry.get("abuseConfidenceScore", 0)

        if score >= 90:
            return ThreatLevel.CRITICAL
        elif score >= 75:
            return ThreatLevel.HIGH
        elif score >= 50:
            return ThreatLevel.MEDIUM
        elif score >= 25:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO


class VirusTotalFeed:
    """VirusTotal threat feed (placeholder)"""

    def __init__(self, config: ThreatIntelConfig):
        self.config = config

    async def fetch_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from VirusTotal"""
        # VirusTotal implementation would go here
        # Requires premium API access for threat feeds
        return []


class CustomFeed:
    """Custom threat feed parser"""

    def __init__(self, feed_url: str, config: ThreatIntelConfig):
        self.feed_url = feed_url
        self.config = config

    async def fetch_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from custom feed"""
        indicators = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.feed_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        indicators = self._parse_feed(content)

        except Exception as e:
            logger.error(f"Error fetching custom feed {self.feed_url}: {e}")

        return indicators

    def _parse_feed(self, content: str) -> List[ThreatIndicator]:
        """Parse custom feed content"""
        indicators = []

        # Try to detect feed format
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try to parse as JSON
            try:
                data = json.loads(line)
                indicator = self._parse_json_line(data)
                if indicator:
                    indicators.append(indicator)
                continue
            except:
                pass

            # Try to parse as CSV/TSV
            parts = re.split(r"[,\t|]", line)
            if len(parts) >= 1:
                # Assume first field is the indicator
                indicator_value = parts[0].strip()

                # Detect indicator type
                ind_type = self._detect_indicator_type(indicator_value)
                if ind_type:
                    indicator = ThreatIndicator(
                        indicator=indicator_value,
                        indicator_type=ind_type,
                        threat_types=[ThreatType.REPUTATION],
                        threat_level=ThreatLevel.MEDIUM,
                        source=urlparse(self.feed_url).netloc or "custom",
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        confidence=0.5,
                    )
                    indicators.append(indicator)

        return indicators

    def _parse_json_line(
        self, data: Dict[str, Any]
    ) -> Optional[ThreatIndicator]:
        """Parse JSON formatted indicator"""
        # Common field mappings
        indicator = (
            data.get("indicator") or data.get("ioc") or data.get("value")
        )
        if not indicator:
            return None

        ind_type = data.get("type") or self._detect_indicator_type(indicator)
        if not ind_type:
            return None

        return ThreatIndicator(
            indicator=indicator,
            indicator_type=ind_type,
            threat_types=[ThreatType.REPUTATION],
            threat_level=ThreatLevel.MEDIUM,
            source=data.get(
                "source", urlparse(self.feed_url).netloc or "custom"
            ),
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=data.get("confidence", 0.5),
            tags=data.get("tags", []),
            metadata=data,
        )

    def _detect_indicator_type(self, value: str) -> Optional[str]:
        """Detect indicator type from value"""
        # IP address
        try:
            ipaddress.ip_address(value)
            return "ip"
        except:
            pass

        # Domain
        if re.match(
            r"^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$", value
        ):
            return "domain"

        # URL
        if value.startswith(("http://", "https://", "ftp://")):
            return "url"

        # Hash (MD5, SHA1, SHA256)
        if re.match(r"^[a-fA-F0-9]{32}$", value):  # MD5
            return "hash"
        elif re.match(r"^[a-fA-F0-9]{40}$", value):  # SHA1
            return "hash"
        elif re.match(r"^[a-fA-F0-9]{64}$", value):  # SHA256
            return "hash"

        return None


class ThreatIntelligenceEngine:
    """Main threat intelligence engine"""

    def __init__(self, config: ThreatIntelConfig):
        self.config = config
        self.feed_manager = ThreatFeedManager(config)
        self.redis_client = None
        self.bloom_filter = BloomFilter(
            max_elements=config.bloom_filter_size,
            error_rate=config.bloom_filter_fp_rate,
        )
        self._rate_limiter = {}

    async def initialize(self) -> None:
        """Initialize the engine"""
        self.redis_client = await redis.from_url(self.config.redis_url)
        await self.load_indicators()

    async def close(self) -> None:
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()

    async def load_indicators(self) -> None:
        """Load threat indicators from feeds"""
        logger.info("Loading threat indicators...")

        # Fetch from all feeds
        indicators = await self.feed_manager.fetch_all_indicators()

        # Store in Redis and Bloom filter
        pipeline = self.redis_client.pipeline()

        for indicator in indicators:
            # Add to Bloom filter for fast checking
            self.bloom_filter.add(indicator.indicator)

            # Store in Redis with expiration
            key = f"threat:{indicator.indicator_type}:{indicator.indicator}"
            value = json.dumps(
                {
                    "threat_types": [t.value for t in indicator.threat_types],
                    "threat_level": indicator.threat_level.value,
                    "source": indicator.source,
                    "confidence": indicator.confidence,
                    "last_seen": indicator.last_seen.isoformat(),
                }
            )

            pipeline.setex(key, self.config.cache_ttl, value)

            # Add to type-specific sets for scanning
            pipeline.sadd(
                f"threats:{indicator.indicator_type}", indicator.indicator
            )

        await pipeline.execute()
        logger.info(f"Loaded {len(indicators)} threat indicators")

    async def check_indicator(
        self, indicator: str, indicator_type: str
    ) -> Optional[Dict[str, Any]]:
        """Check if an indicator is a known threat"""
        # Quick check with Bloom filter
        if indicator not in self.bloom_filter:
            return None

        # Check whitelist
        if indicator_type == "ip" and indicator in self.config.whitelist_ips:
            return None
        elif (
            indicator_type == "domain"
            and indicator in self.config.whitelist_domains
        ):
            return None

        # Get from Redis
        key = f"threat:{indicator_type}:{indicator}"
        data = await self.redis_client.get(key)

        if data:
            threat_info = json.loads(data)
            threat_info["indicator"] = indicator
            threat_info["indicator_type"] = indicator_type
            return threat_info

        return None

    async def check_ip(self, ip: str) -> Optional[Dict[str, Any]]:
        """Check if IP is a threat"""
        return await self.check_indicator(ip, "ip")

    async def check_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """Check if domain is a threat"""
        return await self.check_indicator(domain, "domain")

    async def check_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Check if URL is a threat"""
        # Check full URL
        result = await self.check_indicator(url, "url")
        if result:
            return result

        # Also check domain part
        parsed = urlparse(url)
        if parsed.netloc:
            return await self.check_domain(parsed.netloc)

        return None

    async def check_request(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check request for threats"""
        threats = []

        # Check source IP
        source_ip = request_data.get("source_ip")
        if source_ip:
            threat = await self.check_ip(source_ip)
            if threat:
                threats.append({"type": "source_ip", **threat})

        # Check User-Agent for known bad patterns
        user_agent = request_data.get("user_agent", "")
        if self._is_malicious_user_agent(user_agent):
            threats.append(
                {
                    "type": "user_agent",
                    "indicator": user_agent,
                    "threat_types": ["scanner"],
                    "threat_level": ThreatLevel.MEDIUM.value,
                }
            )

        # Check request path for suspicious patterns
        path = request_data.get("path", "")
        if self._is_suspicious_path(path):
            threats.append(
                {
                    "type": "path",
                    "indicator": path,
                    "threat_types": ["scanner", "exploit"],
                    "threat_level": ThreatLevel.HIGH.value,
                }
            )

        # Check rate limiting
        if await self._check_rate_limit(source_ip):
            threats.append(
                {
                    "type": "rate_limit",
                    "indicator": source_ip,
                    "threat_types": ["brute_force"],
                    "threat_level": ThreatLevel.MEDIUM.value,
                }
            )

        # Determine action
        should_block = False
        highest_threat = ThreatLevel.INFO

        for threat in threats:
            threat_level = ThreatLevel(threat["threat_level"])
            if threat_level.value > highest_threat.value:
                highest_threat = threat_level

            if threat_level.value >= self.config.auto_block_threshold.value:
                should_block = True

        return {
            "threats": threats,
            "should_block": should_block,
            "highest_threat_level": highest_threat.name,
        }

    async def block_indicator(
        self, indicator: str, indicator_type: str, reason: str
    ) -> None:
        """Block an indicator"""
        key = f"blocked:{indicator_type}:{indicator}"
        value = json.dumps(
            {"reason": reason, "blocked_at": datetime.now().isoformat()}
        )

        await self.redis_client.setex(key, self.config.block_duration, value)

        # Add to Bloom filter
        self.bloom_filter.add(indicator)

        logger.warning(f"Blocked {indicator_type} {indicator}: {reason}")

    async def is_blocked(self, indicator: str, indicator_type: str) -> bool:
        """Check if indicator is blocked"""
        key = f"blocked:{indicator_type}:{indicator}"
        result = await self.redis_client.get(key)
        return result is not None

    def _is_malicious_user_agent(self, user_agent: str) -> bool:
        """Check for malicious user agent patterns"""
        malicious_patterns = [
            r"(bot|crawler|spider|scraper)",  # Generic bots
            r"(sqlmap|havij|acunetix|netsparker|nikto)",  # Security scanners
            r"(masscan|nmap|zmap)",  # Port scanners
            r"(wget|curl|python|ruby|perl|java|go-http)",  # Common tools
            r"^$",  # Empty user agent
        ]

        ua_lower = user_agent.lower()
        for pattern in malicious_patterns:
            if re.search(pattern, ua_lower):
                return True

        return False

    def _is_suspicious_path(self, path: str) -> bool:
        """Check for suspicious request paths"""
        suspicious_patterns = [
            r"\.\./",  # Directory traversal
            r"(etc|var)/(passwd|shadow|hosts)",  # System files
            r"\.(git|svn|env|config)",  # Config files
            r"(admin|manager|phpmyadmin|wp-admin)",  # Admin panels
            r"(eval|exec|system|shell)",  # Code execution
            r"<script|javascript:",  # XSS attempts
            r"union.*select|select.*from",  # SQL injection
            r"\${.*}|`.*`",  # Command injection
        ]

        path_lower = path.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, path_lower):
                return True

        return False

    async def _check_rate_limit(self, ip: str) -> bool:
        """Check if IP is rate limited"""
        now = time.time()
        minute_key = int(now / 60)

        # Clean old entries
        self._rate_limiter = {
            k: v
            for k, v in self._rate_limiter.items()
            if k[1] >= minute_key - 1
        }

        # Count requests
        key = (ip, minute_key)
        self._rate_limiter[key] = self._rate_limiter.get(key, 0) + 1

        return self._rate_limiter[key] > self.config.max_requests_per_minute

    async def get_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics"""
        stats = {
            "bloom_filter_size": len(self.bloom_filter),
            "blocked_ips": await self.redis_client.scard("blocked:ip:*"),
            "threat_indicators": {
                "ip": await self.redis_client.scard("threats:ip"),
                "domain": await self.redis_client.scard("threats:domain"),
                "url": await self.redis_client.scard("threats:url"),
                "hash": await self.redis_client.scard("threats:hash"),
            },
            "feeds": len(self.feed_manager.feeds),
        }

        return stats


class ThreatIntelligenceMiddleware:
    """Middleware for FastAPI/Starlette applications"""

    def __init__(self, app, threat_engine: ThreatIntelligenceEngine):
        self.app = app
        self.threat_engine = threat_engine

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract request data
            client_ip = None
            for header in scope.get("headers", []):
                if header[0] == b"x-forwarded-for":
                    client_ip = header[1].decode().split(",")[0].strip()
                    break

            if not client_ip and scope.get("client"):
                client_ip = scope["client"][0]

            # Check if IP is blocked
            if client_ip and await self.threat_engine.is_blocked(
                client_ip, "ip"
            ):
                await self._send_blocked_response(send)
                return

            # Check threat intelligence
            if client_ip:
                request_data = {
                    "source_ip": client_ip,
                    "path": scope.get("path", ""),
                    "user_agent": self._get_header(scope, b"user-agent"),
                }

                check_result = await self.threat_engine.check_request(
                    request_data
                )

                if check_result["should_block"]:
                    # Block the IP
                    await self.threat_engine.block_indicator(
                        client_ip,
                        "ip",
                        f"Threat intelligence: {check_result['highest_threat_level']}",
                    )
                    await self._send_blocked_response(send)
                    return

        await self.app(scope, receive, send)

    def _get_header(self, scope, header_name: bytes) -> Optional[str]:
        """Get header value from scope"""
        for header in scope.get("headers", []):
            if header[0] == header_name:
                return header[1].decode()
        return None

    async def _send_blocked_response(self, send):
        """Send blocked response"""
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {"error": "Access denied", "reason": "Threat detected"}
                ).encode(),
            }
        )


def main():
    """Main entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Threat Intelligence Engine")
    parser.add_argument("--check-ip", help="Check if IP is a threat")
    parser.add_argument("--check-domain", help="Check if domain is a threat")
    parser.add_argument(
        "--load-feeds", action="store_true", help="Load threat feeds"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # Create configuration
    config = ThreatIntelConfig()

    # Create engine
    engine = ThreatIntelligenceEngine(config)

    async def run():
        await engine.initialize()

        if args.load_feeds:
            await engine.load_indicators()
            print("Threat feeds loaded successfully")

        if args.check_ip:
            result = await engine.check_ip(args.check_ip)
            if result:
                print(f"THREAT DETECTED: {json.dumps(result, indent=2)}")
            else:
                print("No threat detected")

        if args.check_domain:
            result = await engine.check_domain(args.check_domain)
            if result:
                print(f"THREAT DETECTED: {json.dumps(result, indent=2)}")
            else:
                print("No threat detected")

        if args.stats:
            stats = await engine.get_statistics()
            print(f"Statistics: {json.dumps(stats, indent=2)}")

        await engine.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
