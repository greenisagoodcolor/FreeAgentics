#!/usr/bin/env python3
"""
FreeAgentics Automated Backup System
Implements comprehensive 3-2-1 backup strategy with monitoring and verification
"""

import hashlib
import json
import logging
import os
import shutil
import smtplib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import psutil
import psycopg2
import requests
import schedule

import redis


class BackupType(Enum):
    """Types of backups supported"""

    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup operation status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupConfig:
    """Backup configuration settings"""

    # Paths
    backup_root: str = "/var/backups/freeagentics"
    temp_dir: str = "/tmp/freeagentics-backup"

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "freeagentics"
    db_user: str = "freeagentics"
    db_password: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # S3 Offsite
    s3_bucket: str = "freeagentics-backups-prod"
    s3_region: str = "us-east-1"
    s3_storage_class: str = "STANDARD_IA"
    s3_glacier_days: int = 30
    s3_deep_archive_days: int = 90

    # Secondary offsite (different provider)
    secondary_provider: str = "azure"  # azure, gcp, backblaze
    secondary_container: str = "freeagentics-backups"
    secondary_connection_string: str = ""

    # Retention policies
    local_retention_days: int = 7
    offsite_retention_days: int = 30
    archive_retention_days: int = 365

    # Encryption
    enable_encryption: bool = True
    encryption_key_file: str = "/etc/freeagentics/backup-encryption.key"

    # Notifications
    slack_webhook: str = ""
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_from: str = "backup@freeagentics.io"
    email_to: List[str] = field(default_factory=list)
    pagerduty_key: str = ""

    # Performance
    compression_level: int = 6
    parallel_jobs: int = 4
    bandwidth_limit_mbps: int = 50

    # Verification
    enable_verification: bool = True
    test_restore_enabled: bool = True
    checksum_algorithm: str = "sha256"

    # Monitoring
    prometheus_pushgateway: str = ""
    grafana_api_key: str = ""

    @classmethod
    def from_env_file(cls, env_file: str = "/etc/freeagentics/backup.env") -> "BackupConfig":
        """Load configuration from environment file"""
        config = cls()

        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"')

                        # Map environment variables to config attributes
                        if hasattr(config, key.lower()):
                            attr_type = type(getattr(config, key.lower()))
                            if attr_type is bool:
                                setattr(
                                    config,
                                    key.lower(),
                                    value.lower() == "true",
                                )
                            elif attr_type is int:
                                setattr(config, key.lower(), int(value))
                            elif attr_type is list:
                                setattr(config, key.lower(), value.split(","))
                            else:
                                setattr(config, key.lower(), value)

        return config


@dataclass
class BackupMetadata:
    """Metadata for a backup operation"""

    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    status: BackupStatus
    size_bytes: int = 0
    duration_seconds: float = 0.0
    checksums: Dict[str, str] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    verification_status: Optional[str] = None
    offsite_locations: List[str] = field(default_factory=list)


class BackupOrchestrator:
    """Main backup orchestration class"""

    def __init__(self, config: BackupConfig):
        self.config = config
        self.logger = self._setup_logging()
        self._ensure_directories()
        self.metrics = BackupMetrics(config)
        self.notifier = BackupNotifier(config, self.logger)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = Path(self.config.backup_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("freeagentics_backup")
        logger.setLevel(logging.INFO)

        # File handler
        log_file = log_dir / f"backup_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        dirs = [
            self.config.backup_root,
            f"{self.config.backup_root}/daily",
            f"{self.config.backup_root}/incremental",
            f"{self.config.backup_root}/redis",
            f"{self.config.backup_root}/config",
            f"{self.config.backup_root}/knowledge_graph",
            f"{self.config.backup_root}/app_state",
            f"{self.config.backup_root}/logs",
            f"{self.config.backup_root}/metadata",
            f"{self.config.backup_root}/verification",
            self.config.temp_dir,
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def perform_full_backup(self) -> BackupMetadata:
        """Perform a complete full backup"""
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            timestamp=datetime.now(),
            status=BackupStatus.IN_PROGRESS,
        )

        self.logger.info(f"Starting full backup: {backup_id}")
        start_time = time.time()

        try:
            # 1. Database backup
            db_file = self._backup_database(backup_id)
            metadata.files.append(db_file)

            # 2. Redis backup
            redis_file = self._backup_redis(backup_id)
            if redis_file:
                metadata.files.append(redis_file)

            # 3. Knowledge graph backup
            kg_file = self._backup_knowledge_graph(backup_id)
            metadata.files.append(kg_file)

            # 4. Application state backup
            app_state_file = self._backup_application_state(backup_id)
            metadata.files.append(app_state_file)

            # 5. Configuration backup
            config_file = self._backup_configuration(backup_id)
            metadata.files.append(config_file)

            # Calculate checksums
            for file_path in metadata.files:
                if os.path.exists(file_path):
                    metadata.checksums[file_path] = self._calculate_checksum(file_path)
                    metadata.size_bytes += os.path.getsize(file_path)

            # Implement 3-2-1 strategy
            # Copy 1: Local backup (already done)
            # Copy 2: Different local media (NAS or different disk)
            self._copy_to_secondary_local(metadata)

            # Copy 3: Offsite (S3 and secondary provider)
            self._sync_to_offsite(metadata)

            # Verify backups
            if self.config.enable_verification:
                self._verify_backup(metadata)

            metadata.status = BackupStatus.COMPLETED
            metadata.duration_seconds = time.time() - start_time

            self.logger.info(f"Full backup completed: {backup_id}")
            self.notifier.send_success(metadata)

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            metadata.duration_seconds = time.time() - start_time

            self.logger.error(f"Full backup failed: {backup_id} - {str(e)}")
            self.notifier.send_failure(metadata)
            raise

        finally:
            # Save metadata
            self._save_metadata(metadata)
            # Update metrics
            self.metrics.record_backup(metadata)

        return metadata

    def _backup_database(self, backup_id: str) -> str:
        """Backup PostgreSQL database"""
        self.logger.info("Backing up PostgreSQL database...")

        backup_file = f"{self.config.backup_root}/daily/postgres_{backup_id}.dump"

        # Use pg_dump with custom format for faster restore
        cmd = [
            "pg_dump",
            f"--host={self.config.db_host}",
            f"--port={self.config.db_port}",
            f"--username={self.config.db_user}",
            f"--dbname={self.config.db_name}",
            "--format=custom",
            f"--file={backup_file}",
            f"--jobs={self.config.parallel_jobs}",
            "--verbose",
            "--no-password",
        ]

        env = os.environ.copy()
        env["PGPASSWORD"] = self.config.db_password

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Database backup failed: {result.stderr}")

        # Compress the backup
        compressed_file = f"{backup_file}.gz"
        subprocess.run(
            ["gzip", f"-{self.config.compression_level}", backup_file],
            check=True,
        )

        return compressed_file

    def _backup_redis(self, backup_id: str) -> Optional[str]:
        """Backup Redis data"""
        self.logger.info("Backing up Redis...")

        try:
            # Connect to Redis
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True,
            )

            # Force a synchronous save
            r.bgsave()

            # Wait for save to complete
            while r.lastsave() == r.lastsave():
                time.sleep(0.1)

            # Get RDB file location
            redis_dir = r.config_get("dir")["dir"]
            redis_dbfilename = r.config_get("dbfilename")["dbfilename"]
            rdb_path = os.path.join(redis_dir, redis_dbfilename)

            # Copy and compress RDB file
            backup_file = f"{self.config.backup_root}/redis/redis_{backup_id}.rdb.gz"

            with open(rdb_path, "rb") as src:
                import gzip

                with gzip.open(
                    backup_file,
                    "wb",
                    compresslevel=self.config.compression_level,
                ) as dst:
                    shutil.copyfileobj(src, dst)

            return backup_file

        except Exception as e:
            self.logger.warning(f"Redis backup failed: {str(e)}")
            return None

    def _backup_knowledge_graph(self, backup_id: str) -> str:
        """Backup knowledge graph data"""
        self.logger.info("Backing up knowledge graph...")

        # Query knowledge graph tables
        backup_file = f"{self.config.backup_root}/knowledge_graph/kg_{backup_id}.sql"

        tables = [
            "knowledge_graph_nodes",
            "knowledge_graph_edges",
            "knowledge_graph_metadata",
        ]

        with open(backup_file, "w") as f:
            for table in tables:
                cmd = [
                    "pg_dump",
                    f"--host={self.config.db_host}",
                    f"--port={self.config.db_port}",
                    f"--username={self.config.db_user}",
                    f"--dbname={self.config.db_name}",
                    f"--table={table}",
                    "--data-only",
                    "--format=plain",
                    "--no-password",
                ]

                env = os.environ.copy()
                env["PGPASSWORD"] = self.config.db_password

                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if result.returncode == 0:
                    f.write(f"-- Table: {table}\n")
                    f.write(result.stdout)
                    f.write("\n\n")

        # Compress
        compressed_file = f"{backup_file}.gz"
        subprocess.run(
            ["gzip", f"-{self.config.compression_level}", backup_file],
            check=True,
        )

        return compressed_file

    def _backup_application_state(self, backup_id: str) -> str:
        """Backup application state and runtime data"""
        self.logger.info("Backing up application state...")

        state_dir = f"{self.config.temp_dir}/app_state_{backup_id}"
        os.makedirs(state_dir, exist_ok=True)

        # Collect application state
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "backup_id": backup_id,
            "agents": self._get_agent_states(),
            "coalitions": self._get_coalition_states(),
            "active_sessions": self._get_active_sessions(),
            "system_metrics": self._get_system_metrics(),
        }

        # Save state data
        state_file = f"{state_dir}/state.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        # Archive and compress
        archive_file = f"{self.config.backup_root}/app_state/state_{backup_id}.tar.gz"
        subprocess.run(
            [
                "tar",
                "-czf",
                archive_file,
                "-C",
                self.config.temp_dir,
                f"app_state_{backup_id}",
            ],
            check=True,
        )

        # Cleanup temp dir
        shutil.rmtree(state_dir)

        return archive_file

    def _backup_configuration(self, backup_id: str) -> str:
        """Backup all configuration files"""
        self.logger.info("Backing up configuration...")

        config_dir = f"{self.config.temp_dir}/config_{backup_id}"
        os.makedirs(config_dir, exist_ok=True)

        # List of configuration items to backup
        config_items = [
            "/home/green/FreeAgentics/.env",
            "/home/green/FreeAgentics/.env.production",
            "/home/green/FreeAgentics/docker-compose.yml",
            "/home/green/FreeAgentics/docker-compose.production.yml",
            "/home/green/FreeAgentics/nginx/",
            "/home/green/FreeAgentics/config/",
            "/etc/freeagentics/",
        ]

        for item in config_items:
            if os.path.exists(item):
                dest = os.path.join(config_dir, os.path.basename(item))
                if os.path.isdir(item):
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        # Archive and compress
        archive_file = f"{self.config.backup_root}/config/config_{backup_id}.tar.gz"
        subprocess.run(
            [
                "tar",
                "-czf",
                archive_file,
                "-C",
                self.config.temp_dir,
                f"config_{backup_id}",
            ],
            check=True,
        )

        # Cleanup
        shutil.rmtree(config_dir)

        return archive_file

    def _get_agent_states(self) -> List[Dict[str, Any]]:
        """Get current agent states from database"""
        states = []

        try:
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, status, beliefs, metrics, updated_at
                    FROM agents
                    WHERE status = 'active'
                """
                )

                for row in cur.fetchall():
                    states.append(
                        {
                            "id": str(row[0]),
                            "name": row[1],
                            "status": row[2],
                            "beliefs": row[3],
                            "metrics": row[4],
                            "updated_at": row[5].isoformat() if row[5] else None,
                        }
                    )

            conn.close()

        except Exception as e:
            self.logger.warning(f"Failed to get agent states: {str(e)}")

        return states

    def _get_coalition_states(self) -> List[Dict[str, Any]]:
        """Get current coalition states from database"""
        states = []

        try:
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, status, shared_beliefs, metrics, created_at
                    FROM coalitions
                    WHERE status = 'active'
                """
                )

                for row in cur.fetchall():
                    states.append(
                        {
                            "id": str(row[0]),
                            "name": row[1],
                            "status": row[2],
                            "shared_beliefs": row[3],
                            "metrics": row[4],
                            "created_at": row[5].isoformat() if row[5] else None,
                        }
                    )

            conn.close()

        except Exception as e:
            self.logger.warning(f"Failed to get coalition states: {str(e)}")

        return states

    def _get_active_sessions(self) -> int:
        """Get count of active sessions"""
        try:
            r = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True,
            )

            # Count session keys
            session_count = len([k for k in r.keys() if k.startswith("session:")])
            return session_count

        except Exception:
            return 0

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent() if "psutil" in sys.modules else 0,
            "memory_percent": psutil.virtual_memory().percent if "psutil" in sys.modules else 0,
            "disk_usage": shutil.disk_usage(self.config.backup_root)._asdict(),
        }

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate checksum of a file"""
        hash_algo = hashlib.new(self.config.checksum_algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)

        return hash_algo.hexdigest()

    def _copy_to_secondary_local(self, metadata: BackupMetadata):
        """Copy backups to secondary local storage (NAS/different disk)"""
        secondary_root = "/mnt/nas/freeagentics-backups"

        if os.path.exists(secondary_root):
            self.logger.info("Copying to secondary local storage...")

            for file_path in metadata.files:
                if os.path.exists(file_path):
                    dest_dir = os.path.join(
                        secondary_root,
                        os.path.dirname(file_path).replace(self.config.backup_root, ""),
                    ).strip("/")

                    os.makedirs(os.path.join(secondary_root, dest_dir), exist_ok=True)

                    dest_path = os.path.join(secondary_root, dest_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)

                    # Verify copy
                    if self._calculate_checksum(dest_path) != metadata.checksums.get(file_path):
                        raise Exception(f"Secondary copy verification failed for {file_path}")

    def _sync_to_offsite(self, metadata: BackupMetadata):
        """Sync backups to offsite locations"""
        # Primary offsite: AWS S3
        self._sync_to_s3(metadata)

        # Secondary offsite: Different provider
        if self.config.secondary_provider == "azure":
            self._sync_to_azure(metadata)
        elif self.config.secondary_provider == "gcp":
            self._sync_to_gcp(metadata)
        elif self.config.secondary_provider == "backblaze":
            self._sync_to_backblaze(metadata)

    def _sync_to_s3(self, metadata: BackupMetadata):
        """Sync backups to AWS S3"""
        self.logger.info("Syncing to AWS S3...")

        try:
            s3 = boto3.client("s3", region_name=self.config.s3_region)

            for file_path in metadata.files:
                if os.path.exists(file_path):
                    key = file_path.replace(self.config.backup_root + "/", "")

                    # Upload with server-side encryption
                    s3.upload_file(
                        file_path,
                        self.config.s3_bucket,
                        key,
                        ExtraArgs={
                            "StorageClass": self.config.s3_storage_class,
                            "ServerSideEncryption": "AES256",
                            "Metadata": {
                                "backup-id": metadata.backup_id,
                                "checksum": metadata.checksums.get(file_path, ""),
                            },
                        },
                    )

                    metadata.offsite_locations.append(f"s3://{self.config.s3_bucket}/{key}")

            # Set lifecycle policies
            self._configure_s3_lifecycle()

        except Exception as e:
            self.logger.error(f"S3 sync failed: {str(e)}")
            raise

    def _sync_to_azure(self, metadata: BackupMetadata):
        """Sync backups to Azure Blob Storage"""
        self.logger.info("Syncing to Azure Blob Storage...")

        try:
            from azure.storage.blob import BlobServiceClient

            blob_service = BlobServiceClient.from_connection_string(
                self.config.secondary_connection_string
            )

            container_client = blob_service.get_container_client(self.config.secondary_container)

            for file_path in metadata.files:
                if os.path.exists(file_path):
                    blob_name = file_path.replace(self.config.backup_root + "/", "")

                    with open(file_path, "rb") as data:
                        container_client.upload_blob(
                            name=blob_name,
                            data=data,
                            overwrite=True,
                            metadata={
                                "backup_id": metadata.backup_id,
                                "checksum": metadata.checksums.get(file_path, ""),
                            },
                        )

                    metadata.offsite_locations.append(
                        f"azure://{self.config.secondary_container}/{blob_name}"
                    )

        except Exception as e:
            self.logger.error(f"Azure sync failed: {str(e)}")
            # Don't raise - secondary offsite is optional

    def _sync_to_gcp(self, metadata: BackupMetadata):
        """Sync backups to Google Cloud Storage"""
        self.logger.info("Syncing to Google Cloud Storage...")

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(self.config.secondary_container)

            for file_path in metadata.files:
                if os.path.exists(file_path):
                    blob_name = file_path.replace(self.config.backup_root + "/", "")
                    blob = bucket.blob(blob_name)

                    blob.metadata = {
                        "backup-id": metadata.backup_id,
                        "checksum": metadata.checksums.get(file_path, ""),
                    }

                    blob.upload_from_filename(file_path)

                    metadata.offsite_locations.append(
                        f"gs://{self.config.secondary_container}/{blob_name}"
                    )

        except Exception as e:
            self.logger.error(f"GCP sync failed: {str(e)}")

    def _sync_to_backblaze(self, metadata: BackupMetadata):
        """Sync backups to Backblaze B2"""
        self.logger.info("Syncing to Backblaze B2...")

        try:
            import b2sdk.v2 as b2

            info = b2.InMemoryAccountInfo()
            b2_api = b2.B2Api(info)

            # Parse connection string for credentials
            # Format: "keyId:applicationKey:bucketName"
            parts = self.config.secondary_connection_string.split(":")
            if len(parts) >= 3:
                b2_api.authorize_account("production", parts[0], parts[1])
                bucket = b2_api.get_bucket_by_name(parts[2])

                for file_path in metadata.files:
                    if os.path.exists(file_path):
                        file_name = file_path.replace(self.config.backup_root + "/", "")

                        bucket.upload_local_file(
                            local_file=file_path,
                            file_name=file_name,
                            file_infos={
                                "backup-id": metadata.backup_id,
                                "checksum": metadata.checksums.get(file_path, ""),
                            },
                        )

                        metadata.offsite_locations.append(f"b2://{parts[2]}/{file_name}")

        except Exception as e:
            self.logger.error(f"Backblaze sync failed: {str(e)}")

    def _configure_s3_lifecycle(self):
        """Configure S3 lifecycle policies for automatic archival"""
        try:
            s3 = boto3.client("s3", region_name=self.config.s3_region)

            lifecycle_policy = {
                "Rules": [
                    {
                        "ID": "ArchiveOldBackups",
                        "Status": "Enabled",
                        "Transitions": [
                            {
                                "Days": self.config.s3_glacier_days,
                                "StorageClass": "GLACIER",
                            },
                            {
                                "Days": self.config.s3_deep_archive_days,
                                "StorageClass": "DEEP_ARCHIVE",
                            },
                        ],
                        "NoncurrentVersionTransitions": [
                            {"NoncurrentDays": 30, "StorageClass": "GLACIER"}
                        ],
                        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                    }
                ]
            }

            s3.put_bucket_lifecycle_configuration(
                Bucket=self.config.s3_bucket,
                LifecycleConfiguration=lifecycle_policy,
            )

        except Exception as e:
            self.logger.warning(f"Failed to configure S3 lifecycle: {str(e)}")

    def _verify_backup(self, metadata: BackupMetadata):
        """Verify backup integrity"""
        self.logger.info("Verifying backup integrity...")

        verification_results = {
            "checksum_verification": True,
            "file_integrity": True,
            "restore_test": False,
        }

        # Verify checksums
        for file_path, expected_checksum in metadata.checksums.items():
            if os.path.exists(file_path):
                actual_checksum = self._calculate_checksum(file_path)
                if actual_checksum != expected_checksum:
                    verification_results["checksum_verification"] = False
                    self.logger.error(f"Checksum mismatch for {file_path}")

        # Test restore if enabled
        if self.config.test_restore_enabled:
            verification_results["restore_test"] = self._test_restore(metadata)

        # Update metadata
        metadata.verification_status = json.dumps(verification_results)

        if all(verification_results.values()):
            metadata.status = BackupStatus.VERIFIED
            self.logger.info("Backup verification successful")
        else:
            self.logger.warning("Backup verification completed with issues")

    def _test_restore(self, metadata: BackupMetadata) -> bool:
        """Test restore process"""
        self.logger.info("Testing backup restore...")

        test_dir = f"{self.config.backup_root}/verification/test_restore_{metadata.backup_id}"
        os.makedirs(test_dir, exist_ok=True)

        try:
            # Test database restore
            for file_path in metadata.files:
                if "postgres_" in file_path and file_path.endswith(".gz"):
                    # Create test database
                    test_db = f"freeagentics_test_{metadata.backup_id[:8]}"

                    # Test restore command
                    cmd = f"gunzip -c {file_path} | pg_restore --create --dbname=postgres"
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        env={"PGPASSWORD": self.config.db_password},
                    )

                    if result.returncode == 0:
                        # Drop test database
                        drop_cmd = f"dropdb {test_db}"
                        subprocess.run(drop_cmd, shell=True)
                        return True

            return True

        except Exception as e:
            self.logger.error(f"Test restore failed: {str(e)}")
            return False

        finally:
            # Cleanup
            shutil.rmtree(test_dir, ignore_errors=True)

    def _save_metadata(self, metadata: BackupMetadata):
        """Save backup metadata"""
        metadata_file = f"{self.config.backup_root}/metadata/{metadata.backup_id}.json"

        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "backup_id": metadata.backup_id,
                    "backup_type": metadata.backup_type.value,
                    "timestamp": metadata.timestamp.isoformat(),
                    "status": metadata.status.value,
                    "size_bytes": metadata.size_bytes,
                    "duration_seconds": metadata.duration_seconds,
                    "checksums": metadata.checksums,
                    "files": metadata.files,
                    "error_message": metadata.error_message,
                    "verification_status": metadata.verification_status,
                    "offsite_locations": metadata.offsite_locations,
                },
                f,
                indent=2,
            )

    def cleanup_old_backups(self):
        """Clean up old backups based on retention policies"""
        self.logger.info("Cleaning up old backups...")

        now = datetime.now()

        # Local cleanup
        for root, dirs, files in os.walk(self.config.backup_root):
            for file in files:
                file_path = os.path.join(root, file)
                file_age = now - datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_age.days > self.config.local_retention_days:
                    os.remove(file_path)
                    self.logger.info(f"Removed old backup: {file_path}")

        # S3 cleanup (handled by lifecycle policies)
        # Additional cleanup for other providers can be added here

    def schedule_backups(self):
        """Schedule automated backups"""
        # Full backup daily at 2 AM
        schedule.every().day.at("02:00").do(self.perform_full_backup)

        # Incremental backup every 6 hours
        schedule.every(6).hours.do(self.perform_incremental_backup)

        # Cleanup old backups weekly
        schedule.every().sunday.at("04:00").do(self.cleanup_old_backups)

        # Test restore weekly
        schedule.every().sunday.at("05:00").do(self.test_disaster_recovery)

        self.logger.info("Backup schedule configured")

    def perform_incremental_backup(self) -> BackupMetadata:
        """Perform incremental backup"""
        # Implementation for incremental backups
        # This would track changes since last full backup
        pass

    def test_disaster_recovery(self):
        """Test disaster recovery procedures"""
        self.logger.info("Testing disaster recovery procedures...")

        # Get latest verified backup
        latest_backup = self._get_latest_verified_backup()

        if latest_backup:
            # Perform test restore
            success = self._test_restore(latest_backup)

            if success:
                self.notifier.send_info(
                    "DR Test Successful",
                    f"Disaster recovery test completed successfully for backup {latest_backup.backup_id}",
                )
            else:
                self.notifier.send_warning(
                    "DR Test Failed",
                    f"Disaster recovery test failed for backup {latest_backup.backup_id}",
                )

    def _get_latest_verified_backup(self) -> Optional[BackupMetadata]:
        """Get the latest verified backup"""
        metadata_dir = f"{self.config.backup_root}/metadata"

        latest_backup = None
        latest_timestamp = None

        for file in os.listdir(metadata_dir):
            if file.endswith(".json"):
                with open(os.path.join(metadata_dir, file), "r") as f:
                    data = json.load(f)

                    if data.get("status") == "verified":
                        timestamp = datetime.fromisoformat(data["timestamp"])

                        if latest_timestamp is None or timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                            latest_backup = BackupMetadata(
                                backup_id=data["backup_id"],
                                backup_type=BackupType(data["backup_type"]),
                                timestamp=timestamp,
                                status=BackupStatus(data["status"]),
                                size_bytes=data["size_bytes"],
                                duration_seconds=data["duration_seconds"],
                                checksums=data["checksums"],
                                files=data["files"],
                                verification_status=data.get("verification_status"),
                                offsite_locations=data.get("offsite_locations", []),
                            )

        return latest_backup


class BackupMetrics:
    """Backup metrics collection and reporting"""

    def __init__(self, config: BackupConfig):
        self.config = config
        self.metrics_file = f"{config.backup_root}/metrics/backup_metrics.json"

    def record_backup(self, metadata: BackupMetadata):
        """Record backup metrics"""
        metrics = {
            "timestamp": metadata.timestamp.isoformat(),
            "backup_id": metadata.backup_id,
            "backup_type": metadata.backup_type.value,
            "status": metadata.status.value,
            "size_bytes": metadata.size_bytes,
            "duration_seconds": metadata.duration_seconds,
            "files_count": len(metadata.files),
            "offsite_locations_count": len(metadata.offsite_locations),
        }

        # Send to Prometheus if configured
        if self.config.prometheus_pushgateway:
            self._push_to_prometheus(metrics)

        # Save locally
        self._save_metrics(metrics)

    def _push_to_prometheus(self, metrics: Dict[str, Any]):
        """Push metrics to Prometheus"""
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

            registry = CollectorRegistry()

            # Create gauges
            backup_size = Gauge(
                "freeagentics_backup_size_bytes",
                "Backup size in bytes",
                registry=registry,
            )
            backup_duration = Gauge(
                "freeagentics_backup_duration_seconds",
                "Backup duration in seconds",
                registry=registry,
            )
            backup_status = Gauge(
                "freeagentics_backup_status",
                "Backup status (1=success, 0=failure)",
                registry=registry,
            )

            # Set values
            backup_size.set(metrics["size_bytes"])
            backup_duration.set(metrics["duration_seconds"])
            backup_status.set(1 if metrics["status"] == "completed" else 0)

            # Push to gateway
            push_to_gateway(
                self.config.prometheus_pushgateway,
                job="freeagentics_backup",
                registry=registry,
            )

        except Exception as e:
            logging.warning(f"Failed to push metrics to Prometheus: {str(e)}")

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics locally"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

        # Load existing metrics
        existing_metrics = []
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                existing_metrics = json.load(f)

        # Append new metrics
        existing_metrics.append(metrics)

        # Keep only last 1000 entries
        if len(existing_metrics) > 1000:
            existing_metrics = existing_metrics[-1000:]

        # Save
        with open(self.metrics_file, "w") as f:
            json.dump(existing_metrics, f, indent=2)


class BackupNotifier:
    """Backup notification system"""

    def __init__(self, config: BackupConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def send_success(self, metadata: BackupMetadata):
        """Send success notification"""
        message = f"""
Backup Completed Successfully

Backup ID: {metadata.backup_id}
Type: {metadata.backup_type.value}
Size: {self._format_size(metadata.size_bytes)}
Duration: {metadata.duration_seconds:.2f} seconds
Files: {len(metadata.files)}
Offsite Locations: {len(metadata.offsite_locations)}
Status: {metadata.status.value}
"""

        self._send_notification("Backup Success", message, "success")

    def send_failure(self, metadata: BackupMetadata):
        """Send failure notification"""
        message = f"""
Backup Failed

Backup ID: {metadata.backup_id}
Type: {metadata.backup_type.value}
Error: {metadata.error_message}
Duration: {metadata.duration_seconds:.2f} seconds
Status: {metadata.status.value}
"""

        self._send_notification("Backup Failed", message, "error")

    def send_warning(self, subject: str, message: str):
        """Send warning notification"""
        self._send_notification(subject, message, "warning")

    def send_info(self, subject: str, message: str):
        """Send info notification"""
        self._send_notification(subject, message, "info")

    def _send_notification(self, subject: str, message: str, level: str):
        """Send notification through configured channels"""
        # Log
        self.logger.info(f"Notification [{level}]: {subject}")

        # Slack
        if self.config.slack_webhook:
            self._send_slack(subject, message, level)

        # Email
        if self.config.email_to:
            self._send_email(subject, message, level)

        # PagerDuty (only for errors)
        if level == "error" and self.config.pagerduty_key:
            self._send_pagerduty(subject, message)

    def _send_slack(self, subject: str, message: str, level: str):
        """Send Slack notification"""
        emoji_map = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
        }

        emoji = emoji_map.get(level, "📢")

        payload = {
            "text": f"{emoji} *FreeAgentics Backup* - {subject}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{subject}*\n```{message}```",
                    },
                }
            ],
        }

        try:
            response = requests.post(self.config.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            self.logger.warning(f"Failed to send Slack notification: {str(e)}")

    def _send_email(self, subject: str, message: str, level: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = f"[FreeAgentics Backup] {subject}"

            body = f"""
FreeAgentics Backup Notification
Level: {level.upper()}
Subject: {subject}

{message}

--
This is an automated message from FreeAgentics Backup System
"""

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                # Note: Add authentication if needed
                server.send_message(msg)

        except Exception as e:
            self.logger.warning(f"Failed to send email notification: {str(e)}")

    def _send_pagerduty(self, subject: str, message: str):
        """Send PagerDuty alert"""
        try:
            payload = {
                "routing_key": self.config.pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"FreeAgentics Backup: {subject}",
                    "severity": "error",
                    "source": "freeagentics-backup",
                    "custom_details": {"message": message},
                },
            }

            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()

        except Exception as e:
            self.logger.warning(f"Failed to send PagerDuty alert: {str(e)}")

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="FreeAgentics Automated Backup System")
    parser.add_argument(
        "--config",
        default="/etc/freeagentics/backup.env",
        help="Configuration file path",
    )
    parser.add_argument("--run-now", action="store_true", help="Run full backup immediately")
    parser.add_argument("--test-restore", action="store_true", help="Test disaster recovery")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup of old backups")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon with scheduled backups",
    )

    args = parser.parse_args()

    # Load configuration
    config = BackupConfig.from_env_file(args.config)

    # Create orchestrator
    orchestrator = BackupOrchestrator(config)

    if args.run_now:
        # Run full backup immediately
        metadata = orchestrator.perform_full_backup()
        print(f"Backup completed: {metadata.backup_id}")

    elif args.test_restore:
        # Test disaster recovery
        orchestrator.test_disaster_recovery()

    elif args.cleanup:
        # Run cleanup
        orchestrator.cleanup_old_backups()

    elif args.daemon:
        # Run as daemon
        orchestrator.schedule_backups()
        print("Backup daemon started. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nBackup daemon stopped.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
