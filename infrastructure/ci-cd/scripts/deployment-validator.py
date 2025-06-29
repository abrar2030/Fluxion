#!/usr/bin/env python3
"""
Fluxion Deployment Validator
Validates deployment readiness and configuration for production environments
"""

import argparse
import json
import os
import sys
import yaml
import boto3
import hvac
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from kubernetes import client, config

class DeploymentValidator:
    """Production deployment readiness validator"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.checks = []
        self.failures = []
        self.warnings = []
        
        # Deployment requirements
        self.requirements = {
            "infrastructure": {
                "kubernetes_cluster": True,
                "load_balancer": True,
                "database": True,
                "cache": True,
                "monitoring": True,
                "logging": True
            },
            "security": {
                "secrets_management": True,
                "network_policies": True,
                "rbac": True,
                "pod_security": True,
                "image_scanning": True
            },
            "compliance": {
                "backup_strategy": True,
                "disaster_recovery": True,
                "audit_logging": True,
                "data_encryption": True
            },
            "performance": {
                "resource_limits": True,
                "auto_scaling": True,
                "health_checks": True,
                "readiness_probes": True
            }
        }
    
    def validate_kubernetes_cluster(self) -> None:
        """Validate Kubernetes cluster readiness"""
        print("🔍 Validating Kubernetes cluster...")
        
        try:
            # Try to load kubeconfig
            config.load_kube_config()
            v1 = client.CoreV1Api()
            
            # Check cluster connectivity
            nodes = v1.list_node()
            if not nodes.items:
                self.failures.append({
                    "category": "kubernetes",
                    "message": "No nodes found in cluster",
                    "severity": "critical"
                })
                return
            
            # Check node readiness
            ready_nodes = 0
            for node in nodes.items:
                for condition in node.status.conditions:
                    if condition.type == "Ready" and condition.status == "True":
                        ready_nodes += 1
                        break
            
            if ready_nodes == 0:
                self.failures.append({
                    "category": "kubernetes",
                    "message": "No ready nodes in cluster",
                    "severity": "critical"
                })
            else:
                self.checks.append(f"Kubernetes cluster ready with {ready_nodes} nodes")
            
            # Check system namespaces
            namespaces = v1.list_namespace()
            system_namespaces = ["kube-system", "kube-public", "kube-node-lease"]
            for ns in system_namespaces:
                if not any(namespace.metadata.name == ns for namespace in namespaces.items):
                    self.failures.append({
                        "category": "kubernetes",
                        "message": f"Missing system namespace: {ns}",
                        "severity": "high"
                    })
            
            # Check for monitoring namespace
            if not any(namespace.metadata.name == "monitoring" for namespace in namespaces.items):
                self.warnings.append({
                    "category": "kubernetes",
                    "message": "Monitoring namespace not found"
                })
            
            # Check for logging namespace
            if not any(namespace.metadata.name == "logging" for namespace in namespaces.items):
                self.warnings.append({
                    "category": "kubernetes",
                    "message": "Logging namespace not found"
                })
                
        except Exception as e:
            self.failures.append({
                "category": "kubernetes",
                "message": f"Cannot connect to Kubernetes cluster: {e}",
                "severity": "critical"
            })
    
    def validate_secrets_management(self) -> None:
        """Validate secrets management system"""
        print("🔍 Validating secrets management...")
        
        vault_addr = os.getenv('VAULT_ADDR')
        vault_token = os.getenv('VAULT_TOKEN')
        
        if not vault_addr:
            self.failures.append({
                "category": "secrets",
                "message": "VAULT_ADDR environment variable not set",
                "severity": "critical"
            })
            return
        
        try:
            # Initialize Vault client
            vault_client = hvac.Client(url=vault_addr, token=vault_token)
            
            # Check Vault connectivity
            if not vault_client.is_authenticated():
                self.failures.append({
                    "category": "secrets",
                    "message": "Cannot authenticate with Vault",
                    "severity": "critical"
                })
                return
            
            self.checks.append("Vault authentication successful")
            
            # Check if Vault is sealed
            if vault_client.sys.is_sealed():
                self.failures.append({
                    "category": "secrets",
                    "message": "Vault is sealed",
                    "severity": "critical"
                })
                return
            
            self.checks.append("Vault is unsealed and ready")
            
            # Check for required secret paths
            required_secrets = [
                "secret/data/jwt",
                "secret/data/api",
                "secret/data/encryption",
                "secret/data/database"
            ]
            
            for secret_path in required_secrets:
                try:
                    response = vault_client.secrets.kv.v2.read_secret_version(path=secret_path.replace('secret/data/', ''))
                    if response:
                        self.checks.append(f"Secret exists: {secret_path}")
                    else:
                        self.failures.append({
                            "category": "secrets",
                            "message": f"Required secret not found: {secret_path}",
                            "severity": "high"
                        })
                except Exception:
                    self.failures.append({
                        "category": "secrets",
                        "message": f"Cannot access secret: {secret_path}",
                        "severity": "high"
                    })
                    
        except Exception as e:
            self.failures.append({
                "category": "secrets",
                "message": f"Vault validation failed: {e}",
                "severity": "critical"
            })
    
    def validate_database_connectivity(self) -> None:
        """Validate database connectivity and configuration"""
        print("🔍 Validating database connectivity...")
        
        # This would typically connect to your actual database
        # For demo purposes, we'll check environment variables
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            self.failures.append({
                "category": "database",
                "message": "DATABASE_URL environment variable not set",
                "severity": "critical"
            })
            return
        
        # Check database URL format
        if not db_url.startswith(('postgresql://', 'mysql://', 'mongodb://')):
            self.failures.append({
                "category": "database",
                "message": "Invalid database URL format",
                "severity": "high"
            })
        else:
            self.checks.append("Database URL format is valid")
        
        # Check for SSL requirement
        if 'sslmode=require' not in db_url and 'ssl=true' not in db_url:
            self.warnings.append({
                "category": "database",
                "message": "Database connection may not enforce SSL"
            })
        else:
            self.checks.append("Database SSL configuration detected")
    
    def validate_monitoring_stack(self) -> None:
        """Validate monitoring and alerting stack"""
        print("🔍 Validating monitoring stack...")
        
        # Check Prometheus
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        try:
            response = requests.get(f"{prometheus_url}/api/v1/status/config", timeout=10)
            if response.status_code == 200:
                self.checks.append("Prometheus is accessible")
            else:
                self.failures.append({
                    "category": "monitoring",
                    "message": f"Prometheus not accessible: {response.status_code}",
                    "severity": "high"
                })
        except Exception as e:
            self.failures.append({
                "category": "monitoring",
                "message": f"Cannot connect to Prometheus: {e}",
                "severity": "high"
            })
        
        # Check Grafana
        grafana_url = os.getenv('GRAFANA_URL', 'http://grafana:3000')
        try:
            response = requests.get(f"{grafana_url}/api/health", timeout=10)
            if response.status_code == 200:
                self.checks.append("Grafana is accessible")
            else:
                self.warnings.append({
                    "category": "monitoring",
                    "message": f"Grafana not accessible: {response.status_code}"
                })
        except Exception as e:
            self.warnings.append({
                "category": "monitoring",
                "message": f"Cannot connect to Grafana: {e}"
            })
    
    def validate_logging_stack(self) -> None:
        """Validate centralized logging stack"""
        print("🔍 Validating logging stack...")
        
        # Check Elasticsearch
        elasticsearch_url = os.getenv('ELASTICSEARCH_URL', 'http://elasticsearch:9200')
        try:
            response = requests.get(f"{elasticsearch_url}/_cluster/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') in ['green', 'yellow']:
                    self.checks.append(f"Elasticsearch cluster health: {health_data.get('status')}")
                else:
                    self.failures.append({
                        "category": "logging",
                        "message": f"Elasticsearch cluster unhealthy: {health_data.get('status')}",
                        "severity": "high"
                    })
            else:
                self.failures.append({
                    "category": "logging",
                    "message": f"Elasticsearch not accessible: {response.status_code}",
                    "severity": "high"
                })
        except Exception as e:
            self.failures.append({
                "category": "logging",
                "message": f"Cannot connect to Elasticsearch: {e}",
                "severity": "high"
            })
        
        # Check Kibana
        kibana_url = os.getenv('KIBANA_URL', 'http://kibana:5601')
        try:
            response = requests.get(f"{kibana_url}/api/status", timeout=10)
            if response.status_code == 200:
                self.checks.append("Kibana is accessible")
            else:
                self.warnings.append({
                    "category": "logging",
                    "message": f"Kibana not accessible: {response.status_code}"
                })
        except Exception as e:
            self.warnings.append({
                "category": "logging",
                "message": f"Cannot connect to Kibana: {e}"
            })
    
    def validate_resource_requirements(self) -> None:
        """Validate resource requirements and limits"""
        print("🔍 Validating resource requirements...")
        
        # Check if running in Kubernetes
        try:
            config.load_kube_config()
            v1 = client.CoreV1Api()
            apps_v1 = client.AppsV1Api()
            
            # Get all deployments
            deployments = apps_v1.list_deployment_for_all_namespaces()
            
            for deployment in deployments.items:
                deployment_name = deployment.metadata.name
                namespace = deployment.metadata.namespace
                
                # Skip system deployments
                if namespace in ['kube-system', 'kube-public', 'kube-node-lease']:
                    continue
                
                containers = deployment.spec.template.spec.containers
                for container in containers:
                    # Check resource limits
                    if not container.resources or not container.resources.limits:
                        self.warnings.append({
                            "category": "resources",
                            "message": f"Container {container.name} in {deployment_name} missing resource limits"
                        })
                    else:
                        self.checks.append(f"Resource limits set for {container.name} in {deployment_name}")
                    
                    # Check resource requests
                    if not container.resources or not container.resources.requests:
                        self.warnings.append({
                            "category": "resources",
                            "message": f"Container {container.name} in {deployment_name} missing resource requests"
                        })
                    
                    # Check health checks
                    if not container.liveness_probe:
                        self.warnings.append({
                            "category": "resources",
                            "message": f"Container {container.name} in {deployment_name} missing liveness probe"
                        })
                    
                    if not container.readiness_probe:
                        self.warnings.append({
                            "category": "resources",
                            "message": f"Container {container.name} in {deployment_name} missing readiness probe"
                        })
                        
        except Exception as e:
            self.warnings.append({
                "category": "resources",
                "message": f"Cannot validate Kubernetes resources: {e}"
            })
    
    def validate_backup_strategy(self) -> None:
        """Validate backup and disaster recovery strategy"""
        print("🔍 Validating backup strategy...")
        
        # Check for backup configuration files
        backup_configs = [
            "infrastructure/scripts/backup.sh",
            "infrastructure/kubernetes/base/backup-cronjob.yaml",
            "infrastructure/terraform/modules/backup"
        ]
        
        backup_found = False
        for config_path in backup_configs:
            if Path(config_path).exists():
                backup_found = True
                self.checks.append(f"Backup configuration found: {config_path}")
                break
        
        if not backup_found:
            self.failures.append({
                "category": "backup",
                "message": "No backup configuration found",
                "severity": "high"
            })
        
        # Check backup retention policy
        backup_retention = os.getenv('BACKUP_RETENTION_DAYS')
        if backup_retention:
            try:
                retention_days = int(backup_retention)
                if retention_days >= 2555:  # 7 years for financial compliance
                    self.checks.append(f"Backup retention meets compliance: {retention_days} days")
                else:
                    self.failures.append({
                        "category": "backup",
                        "message": f"Backup retention insufficient: {retention_days} days (required: 2555)",
                        "severity": "high"
                    })
            except ValueError:
                self.warnings.append({
                    "category": "backup",
                    "message": "Invalid backup retention configuration"
                })
        else:
            self.warnings.append({
                "category": "backup",
                "message": "Backup retention not configured"
            })
    
    def validate_network_security(self) -> None:
        """Validate network security configuration"""
        print("🔍 Validating network security...")
        
        try:
            config.load_kube_config()
            networking_v1 = client.NetworkingV1Api()
            
            # Check for network policies
            network_policies = networking_v1.list_network_policy_for_all_namespaces()
            
            if not network_policies.items:
                self.failures.append({
                    "category": "network",
                    "message": "No network policies found",
                    "severity": "high"
                })
            else:
                self.checks.append(f"Found {len(network_policies.items)} network policies")
            
            # Check for ingress controllers
            v1 = client.CoreV1Api()
            services = v1.list_service_for_all_namespaces()
            
            ingress_found = False
            for service in services.items:
                if 'ingress' in service.metadata.name.lower() or 'nginx' in service.metadata.name.lower():
                    ingress_found = True
                    self.checks.append(f"Ingress controller found: {service.metadata.name}")
                    break
            
            if not ingress_found:
                self.warnings.append({
                    "category": "network",
                    "message": "No ingress controller found"
                })
                
        except Exception as e:
            self.warnings.append({
                "category": "network",
                "message": f"Cannot validate network security: {e}"
            })
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate deployment readiness report"""
        total_checks = len(self.checks) + len(self.warnings) + len(self.failures)
        readiness_score = (len(self.checks) / total_checks * 100) if total_checks > 0 else 0
        
        # Determine deployment readiness
        critical_failures = [f for f in self.failures if f.get('severity') == 'critical']
        deployment_ready = len(critical_failures) == 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "deployment_ready": deployment_ready,
            "readiness_score": round(readiness_score, 2),
            "summary": {
                "total_checks": total_checks,
                "passed": len(self.checks),
                "warnings": len(self.warnings),
                "failures": len(self.failures),
                "critical_failures": len(critical_failures)
            },
            "checks_passed": self.checks,
            "warnings": self.warnings,
            "failures": self.failures,
            "requirements": self.requirements,
            "recommendations": self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Deployment readiness report saved to {output_file}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []
        
        critical_failures = [f for f in self.failures if f.get('severity') == 'critical']
        
        if critical_failures:
            recommendations.append("❌ DEPLOYMENT BLOCKED: Resolve all critical failures before deployment")
            for failure in critical_failures:
                recommendations.append(f"  • {failure['message']}")
        else:
            recommendations.append("✅ No critical issues found - deployment can proceed")
        
        if self.warnings:
            recommendations.append("⚠️  Address warnings to improve deployment reliability:")
            for warning in self.warnings[:3]:  # Show top 3 warnings
                recommendations.append(f"  • {warning['message']}")
        
        if len(self.failures) > len(critical_failures):
            recommendations.append("🔧 Fix high-priority issues for optimal deployment:")
            high_failures = [f for f in self.failures if f.get('severity') == 'high']
            for failure in high_failures[:3]:  # Show top 3 high-priority failures
                recommendations.append(f"  • {failure['message']}")
        
        return recommendations
    
    def print_summary(self) -> None:
        """Print deployment readiness summary"""
        total_checks = len(self.checks) + len(self.warnings) + len(self.failures)
        readiness_score = (len(self.checks) / total_checks * 100) if total_checks > 0 else 0
        critical_failures = [f for f in self.failures if f.get('severity') == 'critical']
        deployment_ready = len(critical_failures) == 0
        
        print("\n" + "="*60)
        print("🚀 DEPLOYMENT READINESS VALIDATION REPORT")
        print("="*60)
        print(f"Environment: {self.environment.upper()}")
        print(f"Readiness Score: {readiness_score:.1f}%")
        print(f"Deployment Status: {'✅ READY' if deployment_ready else '❌ BLOCKED'}")
        print()
        
        print(f"📊 Summary:")
        print(f"  • Total Checks: {total_checks}")
        print(f"  • Passed: {len(self.checks)}")
        print(f"  • Warnings: {len(self.warnings)}")
        print(f"  • Failures: {len(self.failures)}")
        print(f"  • Critical Failures: {len(critical_failures)}")
        print()
        
        if critical_failures:
            print("🚨 Critical Issues (Deployment Blocked):")
            for failure in critical_failures:
                print(f"  🔴 [{failure['category'].upper()}] {failure['message']}")
            print()
        
        if self.failures and len(self.failures) > len(critical_failures):
            print("⚠️  High Priority Issues:")
            high_failures = [f for f in self.failures if f.get('severity') == 'high']
            for failure in high_failures[:5]:  # Show first 5
                print(f"  🟠 [{failure['category'].upper()}] {failure['message']}")
            print()
        
        if self.warnings:
            print("💡 Warnings:")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  🟡 [{warning['category'].upper()}] {warning['message']}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more warnings")
            print()
        
        print("📋 Next Steps:")
        if deployment_ready:
            print("  ✅ All critical requirements met")
            print("  🚀 Deployment can proceed")
            print("  💡 Consider addressing warnings for optimal performance")
        else:
            print("  ❌ Resolve critical failures before deployment")
            print("  🔧 Review infrastructure and security configuration")
            print("  📞 Contact DevOps team if assistance needed")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Fluxion Deployment Validator")
    parser.add_argument("--environment", default="production", help="Target environment")
    parser.add_argument("--validate-secrets", action="store_true", help="Validate secrets management")
    parser.add_argument("--validate-resources", action="store_true", help="Validate resource configuration")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    validator = DeploymentValidator(args.environment)
    
    print("🚀 Starting Deployment Readiness Validation...")
    print(f"🎯 Target environment: {args.environment}")
    print()
    
    # Run core validations
    validator.validate_kubernetes_cluster()
    validator.validate_monitoring_stack()
    validator.validate_logging_stack()
    validator.validate_backup_strategy()
    validator.validate_network_security()
    
    # Run optional validations
    if args.validate_secrets:
        validator.validate_secrets_management()
        validator.validate_database_connectivity()
    
    if args.validate_resources:
        validator.validate_resource_requirements()
    
    # Generate and display report
    report = validator.generate_report(args.output)
    validator.print_summary()
    
    # Exit with appropriate code
    critical_failures = [f for f in validator.failures if f.get('severity') == 'critical']
    sys.exit(0 if len(critical_failures) == 0 else 1)

if __name__ == "__main__":
    main()

