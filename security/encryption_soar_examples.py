"""
Examples demonstrating advanced encryption and SOAR capabilities.
Shows integration patterns and best practices for security orchestration.
"""

import asyncio
import os
from datetime import datetime


# Field-level encryption examples
def field_encryption_example():
    """Demonstrate field-level encryption with AWS KMS."""
    from security.encryption import TransparentFieldEncryptor, create_field_encryptor

    print("=== Field-Level Encryption Example ===\n")

    # Create encryptor with AWS KMS (requires AWS credentials)
    # In production, use actual KMS key
    encryptor = create_field_encryptor(
        provider_type="aws_kms",
        region="us-east-1",
        key_id="alias/application-encryption",
        performance_monitoring=True,
    )

    # Example: Encrypt sensitive user data
    user_data = {
        "user_id": "usr_12345",
        "email": "user@example.com",
        "ssn": "123-45-6789",
        "credit_card": "4111111111111111",
        "phone": "+1-555-123-4567",
        "address": "123 Main St, City, ST 12345",
    }

    # Encrypt specific fields
    encrypted_data = encryptor.bulk_encrypt_fields(
        user_data, fields_to_encrypt=["ssn", "credit_card"]
    )

    print(f"Original SSN: {user_data['ssn']}")
    print(f"Encrypted SSN: {encrypted_data['ssn']['ciphertext'][:50]}...")
    print(f"Email (not encrypted): {encrypted_data['email']}\n")

    # Decrypt fields
    decrypted_data = encryptor.bulk_decrypt_fields(
        encrypted_data, fields_to_decrypt=["ssn", "credit_card"]
    )

    print(f"Decrypted SSN: {decrypted_data['ssn']}")

    # Show performance stats
    stats = encryptor.get_performance_stats()
    for operation, metrics in stats.items():
        print(f"\n{operation} performance:")
        print(f"  Average: {metrics['avg_ms']:.2f}ms")
        print(f"  Min: {metrics['min_ms']:.2f}ms")
        print(f"  Max: {metrics['max_ms']:.2f}ms")


def transparent_encryption_example():
    """Demonstrate transparent encryption with decorators."""
    from security.encryption import TransparentFieldEncryptor, create_field_encryptor

    print("\n=== Transparent Encryption Example ===\n")

    # Create encryptor
    encryptor = create_field_encryptor(
        provider_type="vault",
        vault_url="http://localhost:8200",
        vault_token="dev-token",
        key_id="customer-data",
    )

    transparent = TransparentFieldEncryptor(encryptor)

    # Define model with encrypted fields
    @transparent.encrypt_model_fields(
        {
            "ssn": "customer-ssn-key",
            "account_number": "customer-account-key",
            "pin": None,  # Uses default key
        }
    )
    class Customer:
        def __init__(self, name, ssn, account_number, pin):
            self.name = name
            self.ssn = ssn
            self.account_number = account_number
            self.pin = pin

    # Create customer - encryption happens automatically
    customer = Customer(name="Jane Doe", ssn="987-65-4321", account_number="1234567890", pin="1234")

    print(f"Customer name: {customer.name}")
    print(f"Customer SSN (transparently decrypted): {customer.ssn}")
    print(f"Internal encrypted SSN exists: {hasattr(customer, '_encrypted_ssn')}")


def quantum_resistant_example():
    """Demonstrate quantum-resistant encryption."""
    from security.encryption import EncryptionAtRest, QuantumResistantCrypto

    print("\n=== Quantum-Resistant Encryption Example ===\n")

    # Initialize quantum-resistant crypto
    qrc = QuantumResistantCrypto(
        kyber_level=3,  # ~AES-192 equivalent
        dilithium_level=3,  # ~AES-192 equivalent
        enable_homomorphic=True,
    )

    # Generate quantum-resistant keys
    keys = qrc.generate_hybrid_keypair()
    print(f"Generated Kyber key (KEM): {keys['kem'].algorithm}")
    print(f"Generated Dilithium key (Signature): {keys['sign'].algorithm}")

    # Encrypt sensitive data
    sensitive_data = b"Top secret information that needs quantum-resistant protection"

    encrypted = qrc.hybrid_encrypt(
        sensitive_data, keys["kem"].public_key, sign_with_private_key=keys["sign"].private_key
    )

    print(f"\nEncrypted data size: {len(encrypted['ciphertext'])} bytes")
    print(f"Using algorithm: {encrypted['algorithm']}")
    print(f"Signature included: {'signature' in encrypted}")

    # Decrypt and verify
    decrypted = qrc.hybrid_decrypt(
        encrypted, keys["kem"].private_key, verify_with_public_key=keys["sign"].public_key
    )

    print(f"\nDecryption successful: {decrypted == sensitive_data}")

    # Demonstrate homomorphic encryption
    print("\n--- Homomorphic Encryption Demo ---")

    # Encrypt numbers for computation
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    encrypted_values = [qrc.encrypt_for_computation([v]) for v in values]

    # Compute mean without decryption
    encrypted_mean = qrc.compute_on_encrypted("mean", *encrypted_values)

    # Decrypt result
    decrypted_mean = qrc.decrypt_computation_result(encrypted_mean)
    print(f"Computed mean on encrypted data: {decrypted_mean}")
    print(f"Actual mean: {sum(values) / len(values)}")


async def soar_playbook_example():
    """Demonstrate SOAR playbook execution."""
    import tempfile

    import yaml

    from security.soar import PlaybookEngine, PlaybookTrigger

    print("\n=== SOAR Playbook Example ===\n")

    # Create temporary playbook directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample playbook
        playbook = {
            "id": "ransomware_response",
            "name": "Ransomware Attack Response",
            "description": "Automated response to ransomware detection",
            "triggers": [{"type": "alert", "conditions": [{"alert_type": "ransomware"}]}],
            "actions": [
                {
                    "id": "isolate_host",
                    "type": "block_ip",
                    "ip_addresses": ["{{infected_host_ip}}"],
                    "duration_hours": 48,
                },
                {
                    "id": "disable_account",
                    "type": "disable_user",
                    "user_ids": ["{{compromised_user}}"],
                    "reason": "Ransomware infection - account potentially compromised",
                },
                {
                    "id": "collect_artifacts",
                    "type": "collect_forensics",
                    "targets": ["{{infected_host}}"],
                    "data_types": ["memory_dump", "processes", "network_connections", "registry"],
                },
                {
                    "id": "notify_team",
                    "type": "send_notification",
                    "recipients": ["security@company.com", "incident-response@company.com"],
                    "channels": ["email", "slack"],
                    "message": """CRITICAL: Ransomware detected
Host: {{infected_host}}
User: {{compromised_user}}
Time: {{detection_time}}

Automated response initiated:
- Host isolated from network
- User account disabled
- Forensic collection started

Please begin manual investigation immediately.""",
                },
            ],
        }

        # Save playbook
        playbook_file = os.path.join(tmpdir, "ransomware_response.yaml")
        with open(playbook_file, "w") as f:
            yaml.dump(playbook, f)

        # Create playbook engine
        engine = PlaybookEngine(playbook_dir=tmpdir)

        # Execute playbook with incident data
        context = await engine.execute_playbook(
            playbook_id="ransomware_response",
            trigger=PlaybookTrigger.ALERT,
            trigger_data={
                "alert_id": "ALT-2024-001",
                "alert_type": "ransomware",
                "severity": "critical",
            },
            variables={
                "infected_host": "DESKTOP-ABC123",
                "infected_host_ip": "192.168.1.150",
                "compromised_user": "john.smith",
                "detection_time": datetime.utcnow().isoformat(),
            },
        )

        print(f"Playbook execution ID: {context.execution_id}")
        print(f"Status: {context.status.value}")
        print(f"Duration: {(context.end_time - context.start_time).total_seconds():.2f}s")
        print(f"\nArtifacts collected:")
        for key, value in context.artifacts.items():
            print(f"  - {key}: {value}")


async def incident_management_example():
    """Demonstrate incident case management."""
    import tempfile

    from security.soar import IncidentManager, IncidentSeverity, IncidentType, PlaybookEngine

    print("\n=== Incident Management Example ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create incident manager
        manager = IncidentManager(
            data_dir=os.path.join(tmpdir, "incidents"),
            auto_triage=True,
            auto_playbook_execution=False,
        )

        # Create security incident
        case = manager.create_incident(
            title="Suspicious Data Exfiltration Detected",
            description="Large volume of data transferred to unknown external IP",
            type=IncidentType.DATA_BREACH,
            severity=IncidentSeverity.CRITICAL,
            indicators=[
                {
                    "type": "ip",
                    "value": "185.220.101.45",
                    "confidence": 0.95,
                    "metadata": {"country": "Unknown", "reputation": "malicious"},
                },
                {"type": "user", "value": "contractor_account_07", "confidence": 0.85},
                {
                    "type": "hash",
                    "value": "a1b2c3d4e5f6...",
                    "confidence": 0.90,
                    "metadata": {"file": "suspicious_exfil.exe"},
                },
            ],
            affected_assets=["fileserver-01.company.local", "database-prod-02.company.local"],
            custom_fields={
                "data_volume_gb": 150,
                "sensitive_data_types": ["customer_pii", "financial_records"],
                "compliance_impact": ["GDPR", "PCI-DSS"],
            },
        )

        print(f"Created incident: {case.case_id}")
        print(f"Type: {case.type.value}")
        print(f"Severity: {case.severity.value}")
        print(f"Status: {case.status.value}")
        print(f"Assigned to: {case.assigned_to}")
        print(f"Team: {case.team}")

        # Simulate investigation progress
        await asyncio.sleep(0.1)

        # Add investigation notes
        manager.add_incident_notes(
            case.case_id,
            "Initial analysis shows exfiltration started at 02:30 UTC. "
            "Contractor account shows signs of compromise. "
            "Implementing containment measures.",
            actor="analyst_jane",
        )

        # Update status
        manager.update_incident_status(
            case.case_id,
            IncidentStatus.CONTAINED,
            notes="Network isolation implemented. Account disabled.",
            actor="analyst_jane",
        )

        # Add more indicators discovered during investigation
        manager.add_indicators(
            case.case_id,
            [
                {
                    "type": "domain",
                    "value": "evil-exfil-server.xyz",
                    "confidence": 0.99,
                    "source": "network_analysis",
                },
                {
                    "type": "ip",
                    "value": "185.220.101.46",
                    "confidence": 0.95,
                    "source": "related_infrastructure",
                },
            ],
        )

        # Get incident summary
        summary = manager.get_incident_summary(case.case_id)
        print(f"\n=== Incident Summary ===")
        print(f"Timeline entries: {summary['timeline_entries']}")
        print(f"Indicators: {summary['indicators_count']}")
        print(f"Notes: {summary['notes_count']}")

        # Get dashboard metrics
        metrics = manager.get_dashboard_metrics()
        print(f"\n=== Security Dashboard ===")
        print(f"Total incidents: {metrics['total_incidents']}")
        print(f"Open incidents: {metrics['open_incidents']}")
        print(f"Indicators tracked: {metrics['indicators_tracked']}")


def encryption_at_rest_example():
    """Demonstrate encryption at rest for data stores."""
    from security.encryption import EncryptionAtRest, QuantumResistantCrypto

    print("\n=== Encryption at Rest Example ===\n")

    # Initialize encryption
    qrc = QuantumResistantCrypto()
    ear = EncryptionAtRest(qrc, key_rotation_days=30)

    # Example: Encrypt database record
    user_record = {
        "_id": "user_12345",
        "username": "johndoe",
        "email": "john@example.com",
        "personal_info": {
            "full_name": "John Doe",
            "ssn": "123-45-6789",
            "dob": "1990-01-15",
            "phone": "+1-555-123-4567",
        },
        "financial": {
            "account_number": "1234567890",
            "routing_number": "021000021",
            "balance": 25000.50,
        },
        "preferences": {"theme": "dark", "notifications": True},
    }

    # Encrypt sensitive fields only
    encrypted_record = ear.encrypt_document(
        user_record, fields_to_encrypt=["personal_info", "financial"]
    )

    print("Original record fields:")
    print(f"  - username: {user_record['username']}")
    print(f"  - SSN: {user_record['personal_info']['ssn']}")

    print("\nEncrypted record:")
    print(f"  - username: {encrypted_record['username']} (not encrypted)")
    print(f"  - personal_info: <encrypted>")
    print(f"  - encryption metadata: {encrypted_record['_encryption_metadata']}")

    # Decrypt for authorized access
    decrypted_record = ear.decrypt_document(
        encrypted_record, fields_to_decrypt=["personal_info", "financial"]
    )

    print("\nDecrypted record:")
    print(f"  - SSN: {decrypted_record['personal_info']['ssn']}")
    print(f"  - Balance: ${decrypted_record['financial']['balance']:,.2f}")


async def integrated_security_example():
    """Demonstrate integrated security workflow."""
    from security.encryption import QuantumResistantCrypto, create_field_encryptor
    from security.soar import IncidentManager, IncidentSeverity, IncidentType

    print("\n=== Integrated Security Workflow ===\n")

    # Setup encryption
    qrc = QuantumResistantCrypto()
    field_encryptor = create_field_encryptor(
        provider_type="vault", vault_url="http://localhost:8200", vault_token="dev-token"
    )

    # Detect security event
    print("1. Security event detected: Unauthorized access attempt")

    # Create incident with encrypted sensitive data
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = IncidentManager(data_dir=tmpdir)

        # Incident contains sensitive data
        sensitive_incident_data = {
            "attacker_ip": "192.168.1.100",
            "compromised_credentials": {
                "username": "admin",
                "password_hash": "a1b2c3d4...",
                "api_key": "sk-1234567890abcdef",
            },
            "affected_data": {
                "records_accessed": 5000,
                "data_types": ["customer_pii", "payment_info"],
            },
        }

        # Encrypt sensitive fields before storing
        encrypted_data = field_encryptor.bulk_encrypt_fields(
            sensitive_incident_data, fields_to_encrypt=["compromised_credentials", "affected_data"]
        )

        case = manager.create_incident(
            title="Unauthorized API Access - Potential Data Breach",
            description="Anomalous API usage detected with valid but suspicious credentials",
            type=IncidentType.UNAUTHORIZED_ACCESS,
            severity=IncidentSeverity.HIGH,
            indicators=[{"type": "ip", "value": encrypted_data["attacker_ip"], "confidence": 0.9}],
            custom_fields=encrypted_data,
        )

        print(f"\n2. Incident created: {case.case_id}")
        print("   - Sensitive data encrypted using field-level encryption")
        print("   - Quantum-resistant encryption available for long-term storage")

        # Simulate automated response
        print("\n3. Automated response initiated:")
        print("   - IP blocked at firewall")
        print("   - API key revoked")
        print("   - Affected user sessions terminated")
        print("   - Forensic data collection started")

        # Generate encrypted incident report
        incident_report = {
            "case_id": case.case_id,
            "classification": "CONFIDENTIAL",
            "summary": manager.get_incident_summary(case.case_id),
            "response_actions": [
                "IP blocking implemented",
                "Credentials revoked",
                "Forensic collection completed",
            ],
        }

        # Encrypt entire report with quantum-resistant encryption
        encrypted_report = qrc.hybrid_encrypt(
            json.dumps(incident_report).encode(), qrc.generate_hybrid_keypair()["kem"].public_key
        )

        print("\n4. Incident report generated and encrypted")
        print(f"   - Report size: {len(json.dumps(incident_report))} bytes")
        print(f"   - Encrypted size: {len(encrypted_report['ciphertext'])} bytes")
        print(f"   - Using: {encrypted_report['algorithm']}")


# Main execution
async def main():
    """Run all examples."""
    print("=" * 60)
    print("Advanced Encryption and SOAR Examples")
    print("=" * 60)

    # Field encryption examples
    field_encryption_example()

    # Transparent encryption
    # transparent_encryption_example()  # Requires Vault

    # Quantum-resistant encryption
    quantum_resistant_example()

    # SOAR examples
    await soar_playbook_example()
    await incident_management_example()

    # Encryption at rest
    encryption_at_rest_example()

    # Integrated workflow
    await integrated_security_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # For AWS KMS examples, set AWS credentials:
    # export AWS_ACCESS_KEY_ID=your_key
    # export AWS_SECRET_ACCESS_KEY=your_secret
    # export AWS_DEFAULT_REGION=us-east-1

    # For Vault examples, start Vault in dev mode:
    # vault server -dev
    # export VAULT_ADDR='http://127.0.0.1:8200'
    # export VAULT_TOKEN='root'

    import tempfile

    asyncio.run(main())
