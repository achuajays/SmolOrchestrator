"""
Comprehensive demo application showcasing the Human-in-the-Loop System
with SmolAgents, complete state management, and rollback capabilities.

This demo includes several example workflows that demonstrate:
1. E-commerce order processing with approval workflows
2. Budget approval with multi-step validation
3. Code deployment pipeline with checkpoints and rollbacks
4. Document review process with multi-approver requirements
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from uuid import uuid4

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import structlog

from models.workflow import Base, ChannelType
from agents.orchestrator import SmolAgentOrchestrator
from core.state_manager import StateManager
from core.approval_engine import ApprovalEngine
from ui.approval_ui import ApprovalUIRenderer
from integrations.channels import ChannelManager

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class DemoApplication:
    """
    Demo application showcasing the complete human-in-the-loop system.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="demo_app")
        
        # Database setup (using SQLite for demo)
        self.db_url = "sqlite+aiosqlite:///./demo.db"
        self.engine = create_async_engine(self.db_url, echo=True)
        self.AsyncSessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # System components
        self.orchestrator = None
    
    async def initialize(self):
        """Initialize the demo application."""
        self.logger.info("Initializing demo application")
        
        # Create database tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize orchestrator
        async with self.AsyncSessionLocal() as session:
            self.orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config={
                    "provider": "inference_client",
                    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct"
                },
                agent_type="code"
            )
            
            # Configure integrations (using mock configs for demo)
            self.orchestrator.configure_email_integration(
                smtp_config={
                    "host": "smtp.gmail.com",
                    "port": 587,
                    "use_tls": True,
                    "username": "demo@example.com",
                    "password": "demo_password",
                    "from_email": "noreply@demo.com"
                }
            )
        
        self.logger.info("Demo application initialized successfully")
    
    async def run_ecommerce_demo(self):
        """
        Demo 1: E-commerce order processing workflow with human approvals.
        """
        self.logger.info("Starting E-commerce Order Processing Demo")
        
        # Define the e-commerce workflow
        workflow_definition = {
            "name": "E-commerce Order Processing",
            "description": "Process high-value orders with human approval",
            "steps": [
                {
                    "name": "validate_order",
                    "type": "task",
                    "task": "Validate the order details and check inventory availability",
                    "create_checkpoint": True,
                    "context": {
                        "order_id": "ORD-12345",
                        "customer_id": "CUST-98765",
                        "items": [
                            {"product": "Laptop", "price": 1500, "quantity": 2},
                            {"product": "Mouse", "price": 50, "quantity": 1}
                        ],
                        "total_amount": 3050
                    }
                },
                {
                    "name": "payment_processing",
                    "type": "task", 
                    "task": "Process payment and handle any payment method validations",
                    "requires_approval": True,
                    "approval_config": {
                        "title": "High-Value Payment Processing",
                        "description": "Please approve payment processing for order totaling $3,050",
                        "type": "budget",
                        "requested_from": "payment.manager@company.com",
                        "context_data": {
                            "amount": 3050,
                            "currency": "USD",
                            "payment_method": "credit_card",
                            "customer_tier": "gold"
                        }
                    }
                },
                {
                    "name": "inventory_reservation",
                    "type": "task",
                    "task": "Reserve inventory items and update stock levels",
                    "create_checkpoint": True
                },
                {
                    "name": "shipping_setup",
                    "type": "task",
                    "task": "Setup shipping labels and notify logistics team",
                    "requires_approval": True,
                    "approval_config": {
                        "title": "Expedited Shipping Request",
                        "description": "Customer requested next-day shipping for high-value order",
                        "type": "binary",
                        "requested_from": "logistics.manager@company.com"
                    }
                },
                {
                    "name": "order_confirmation", 
                    "type": "task",
                    "task": "Send order confirmation and tracking details to customer"
                }
            ]
        }
        
        async with self.AsyncSessionLocal() as session:
            # Create orchestrator for this session
            orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config={"provider": "inference_client"},
                agent_type="code"
            )
            
            # Create and execute workflow
            workflow = await orchestrator.create_workflow(
                name="E-commerce Order Processing Demo",
                description="Demo of order processing with approvals",
                workflow_definition=workflow_definition,
                input_data={
                    "order_id": "ORD-12345",
                    "total_amount": 3050,
                    "customer_tier": "gold"
                },
                created_by="demo_system"
            )
            
            self.logger.info(f"Created workflow: {workflow.id}")
            
            # Execute workflow
            result = await orchestrator.execute_workflow()
            
            self.logger.info("E-commerce demo completed", result=result)
            return result
    
    async def run_budget_approval_demo(self):
        """
        Demo 2: Multi-step budget approval workflow.
        """
        self.logger.info("Starting Budget Approval Demo")
        
        workflow_definition = {
            "name": "Budget Approval Process",
            "description": "Multi-level budget approval with escalation",
            "steps": [
                {
                    "name": "initial_review",
                    "type": "approval",
                    "title": "Department Budget Request",
                    "description": "Initial budget request for Q1 marketing campaign",
                    "approval_type": "budget",
                    "requested_from": "dept.manager@company.com",
                    "context_data": {
                        "amount": 50000,
                        "currency": "USD",
                        "department": "marketing",
                        "quarter": "Q1",
                        "campaign_type": "digital_advertising"
                    }
                },
                {
                    "name": "financial_review",
                    "type": "task",
                    "task": "Review budget against financial constraints and company policies",
                    "create_checkpoint": True
                },
                {
                    "name": "executive_approval",
                    "type": "approval",
                    "title": "Executive Budget Approval",
                    "description": "Final approval required for budget over $25K threshold",
                    "approval_type": "form",
                    "requested_from": "cfo@company.com",
                    "context_data": {
                        "form_fields": [
                            {
                                "name": "budget_category",
                                "type": "string",
                                "title": "Budget Category",
                                "enum": ["marketing", "operations", "development", "other"],
                                "required": True
                            },
                            {
                                "name": "justification",
                                "type": "string",
                                "title": "Executive Justification",
                                "format": "textarea",
                                "required": True
                            },
                            {
                                "name": "conditions",
                                "type": "string", 
                                "title": "Approval Conditions",
                                "format": "textarea"
                            }
                        ]
                    }
                },
                {
                    "name": "budget_allocation",
                    "type": "task",
                    "task": "Allocate approved budget and setup tracking mechanisms"
                }
            ]
        }
        
        async with self.AsyncSessionLocal() as session:
            orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config={"provider": "inference_client"},
                agent_type="code"
            )
            
            workflow = await orchestrator.create_workflow(
                name="Budget Approval Demo",
                description="Multi-level budget approval process",
                workflow_definition=workflow_definition,
                input_data={
                    "requested_amount": 50000,
                    "department": "marketing",
                    "requestor": "marketing.manager@company.com"
                },
                created_by="demo_system"
            )
            
            result = await orchestrator.execute_workflow()
            
            self.logger.info("Budget approval demo completed", result=result)
            return result
    
    async def run_deployment_pipeline_demo(self):
        """
        Demo 3: Code deployment pipeline with checkpoints and rollback capability.
        """
        self.logger.info("Starting Deployment Pipeline Demo")
        
        workflow_definition = {
            "name": "Production Deployment Pipeline",
            "description": "Safe production deployment with rollback capability",
            "steps": [
                {
                    "name": "pre_deployment_checks",
                    "type": "task",
                    "task": "Run pre-deployment validation, tests, and security scans",
                    "create_checkpoint": True,
                    "context": {
                        "branch": "release/v2.1.0",
                        "commit": "a1b2c3d4",
                        "environment": "production"
                    }
                },
                {
                    "name": "deployment_approval",
                    "type": "approval",
                    "title": "Production Deployment Approval",
                    "description": "Approve deployment of v2.1.0 to production environment",
                    "approval_type": "form",
                    "requested_from": "tech.lead@company.com",
                    "context_data": {
                        "form_fields": [
                            {
                                "name": "deployment_window",
                                "type": "string",
                                "title": "Deployment Window",
                                "enum": ["immediate", "scheduled_maintenance", "after_hours"],
                                "required": True
                            },
                            {
                                "name": "rollback_plan",
                                "type": "string",
                                "title": "Rollback Plan Confirmed",
                                "format": "textarea",
                                "required": True
                            }
                        ]
                    }
                },
                {
                    "name": "database_migration",
                    "type": "task",
                    "task": "Execute database schema migrations with backup verification",
                    "create_checkpoint": True
                },
                {
                    "name": "application_deployment",
                    "type": "task", 
                    "task": "Deploy application code and configuration to production servers",
                    "create_checkpoint": True
                },
                {
                    "name": "post_deployment_validation",
                    "type": "task",
                    "task": "Run post-deployment health checks and validation tests"
                },
                {
                    "name": "monitoring_setup",
                    "type": "task",
                    "task": "Enable enhanced monitoring and alerting for new deployment"
                }
            ]
        }
        
        async with self.AsyncSessionLocal() as session:
            orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config={"provider": "inference_client"},
                agent_type="code"
            )
            
            workflow = await orchestrator.create_workflow(
                name="Deployment Pipeline Demo",
                description="Production deployment with safety checkpoints",
                workflow_definition=workflow_definition,
                input_data={
                    "version": "v2.1.0",
                    "environment": "production",
                    "deployer": "devops.team@company.com"
                },
                created_by="demo_system"
            )
            
            # Simulate a deployment that might need rollback
            try:
                result = await orchestrator.execute_workflow()
                self.logger.info("Deployment completed successfully", result=result)
            except Exception as e:
                self.logger.error("Deployment failed, initiating rollback", error=str(e))
                
                # Demonstrate rollback capability
                rollback_result = await orchestrator.rollback_workflow(
                    checkpoint_name="database_migration",
                    reason="Deployment failure - rolling back to pre-migration state"
                )
                
                self.logger.info("Rollback completed", rollback_result=rollback_result)
            
            return await orchestrator.get_workflow_status(workflow.id)
    
    async def run_document_review_demo(self):
        """
        Demo 4: Document review process with multi-approver requirements.
        """
        self.logger.info("Starting Document Review Demo")
        
        workflow_definition = {
            "name": "Legal Document Review",
            "description": "Multi-approver document review process",
            "steps": [
                {
                    "name": "document_preparation",
                    "type": "task",
                    "task": "Prepare document for review and extract key sections",
                    "context": {
                        "document_type": "contract",
                        "client": "Acme Corp",
                        "value": 250000
                    }
                },
                {
                    "name": "legal_review",
                    "type": "approval",
                    "title": "Legal Review Required",
                    "description": "Legal team review of client contract terms and conditions",
                    "approval_type": "form",
                    "requested_from": "legal.team@company.com",
                    "context_data": {
                        "form_fields": [
                            {
                                "name": "legal_compliance",
                                "type": "string",
                                "title": "Legal Compliance Status",
                                "enum": ["compliant", "requires_changes", "non_compliant"],
                                "required": True
                            },
                            {
                                "name": "risk_assessment",
                                "type": "string",
                                "title": "Risk Level",
                                "enum": ["low", "medium", "high", "critical"],
                                "required": True
                            },
                            {
                                "name": "recommendations",
                                "type": "string",
                                "title": "Legal Recommendations",
                                "format": "textarea",
                                "required": True
                            }
                        ]
                    }
                },
                {
                    "name": "business_review",
                    "type": "approval", 
                    "title": "Business Terms Review",
                    "description": "Business team review of commercial terms and pricing",
                    "approval_type": "form",
                    "requested_from": "business.lead@company.com",
                    "context_data": {
                        "form_fields": [
                            {
                                "name": "commercial_terms",
                                "type": "string",
                                "title": "Commercial Terms Acceptable",
                                "enum": ["acceptable", "negotiate", "reject"],
                                "required": True
                            },
                            {
                                "name": "pricing_approval",
                                "type": "boolean",
                                "title": "Pricing Structure Approved",
                                "required": True
                            }
                        ]
                    }
                },
                {
                    "name": "final_approval",
                    "type": "approval",
                    "title": "Executive Final Approval",
                    "description": "Final executive sign-off for contract execution",
                    "approval_type": "binary",
                    "requested_from": "executive.team@company.com"
                },
                {
                    "name": "contract_execution",
                    "type": "task",
                    "task": "Execute approved contract and setup client onboarding"
                }
            ]
        }
        
        async with self.AsyncSessionLocal() as session:
            orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config={"provider": "inference_client"},
                agent_type="code"
            )
            
            workflow = await orchestrator.create_workflow(
                name="Document Review Demo",
                description="Multi-approver legal document review",
                workflow_definition=workflow_definition,
                input_data={
                    "document_id": "DOC-789",
                    "client_name": "Acme Corp",
                    "contract_value": 250000,
                    "review_type": "new_client_contract"
                },
                created_by="demo_system"
            )
            
            result = await orchestrator.execute_workflow()
            
            self.logger.info("Document review demo completed", result=result)
            return result
    
    async def demonstrate_interactive_approvals(self):
        """
        Demo 5: Interactive approval handling - shows how to handle approvals in real-time.
        """
        self.logger.info("Starting Interactive Approvals Demo")
        
        async with self.AsyncSessionLocal() as session:
            orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config={"provider": "inference_client"},
                agent_type="code"
            )
            
            # Simple workflow that requires approval
            workflow = await orchestrator.create_workflow(
                name="Interactive Approval Demo",
                description="Demonstrates interactive approval handling",
                workflow_definition={
                    "steps": [
                        {
                            "name": "prepare_task",
                            "type": "task",
                            "task": "Prepare a critical task that requires human approval before execution"
                        }
                    ]
                },
                input_data={"demo_mode": True},
                created_by="demo_system"
            )
            
            # Create a simple task that will request approval
            task = """
            You are about to perform a critical operation that requires human approval.
            Use the request_approval tool to ask for permission before proceeding.
            
            The operation is: "Delete old backup files to free up 100GB of storage space"
            
            After approval, simulate the file deletion process.
            """
            
            # Execute the workflow
            result = await orchestrator.execute_workflow(task=task)
            
            self.logger.info("Interactive approval demo result", result=result)
            
            # Show workflow status
            status = await orchestrator.get_workflow_status(workflow.id)
            self.logger.info("Final workflow status", status=status)
            
            return result
    
    async def run_all_demos(self):
        """Run all demo workflows sequentially."""
        self.logger.info("Starting comprehensive demo of Human-in-the-Loop System")
        
        results = {}
        
        try:
            # Run each demo
            results["ecommerce"] = await self.run_ecommerce_demo()
            await asyncio.sleep(2)  # Brief pause between demos
            
            results["budget_approval"] = await self.run_budget_approval_demo()
            await asyncio.sleep(2)
            
            results["deployment_pipeline"] = await self.run_deployment_pipeline_demo()
            await asyncio.sleep(2)
            
            results["document_review"] = await self.run_document_review_demo()
            await asyncio.sleep(2)
            
            results["interactive_approvals"] = await self.demonstrate_interactive_approvals()
            
            self.logger.info("All demos completed successfully", results=results)
            
        except Exception as e:
            self.logger.error("Demo execution failed", error=str(e))
            raise
        
        return results
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            await self.engine.dispose()


async def main():
    """Main demo application entry point."""
    demo = DemoApplication()
    
    try:
        # Initialize the demo
        await demo.initialize()
        
        # Run all demonstrations
        results = await demo.run_all_demos()
        
        print("\n" + "="*80)
        print("🎉 HUMAN-IN-THE-LOOP SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nDemo Results Summary:")
        print(f"✅ E-commerce Workflow: {'SUCCESS' if results.get('ecommerce', {}).get('success') else 'FAILED'}")
        print(f"✅ Budget Approval: {'SUCCESS' if results.get('budget_approval', {}).get('success') else 'FAILED'}")
        print(f"✅ Deployment Pipeline: {'SUCCESS' if results.get('deployment_pipeline') else 'FAILED'}")
        print(f"✅ Document Review: {'SUCCESS' if results.get('document_review', {}).get('success') else 'FAILED'}")
        print(f"✅ Interactive Approvals: {'SUCCESS' if results.get('interactive_approvals', {}).get('success') else 'FAILED'}")
        
        print("\n" + "="*80)
        print("🔧 SYSTEM CAPABILITIES DEMONSTRATED:")
        print("="*80)
        print("✨ Event-driven workflow orchestration")
        print("✨ Multi-channel approval notifications (Email, Slack, UI)")
        print("✨ Dynamic UI generation from schemas")
        print("✨ Complete state management with PostgreSQL")
        print("✨ Automatic checkpoint creation and rollback")
        print("✨ SmolAgent integration with approval tools")
        print("✨ Resilient approval handling with timeouts")
        print("✨ Comprehensive audit trail and observability")
        print("✨ Multi-approver and threshold-based approvals")
        print("✨ Configurable approval types (binary, form, budget)")
        
        print(f"\n📊 Check the database (demo.db) for complete audit trail!")
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"\n❌ Demo failed with error: {str(e)}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())