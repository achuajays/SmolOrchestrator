"""
Main application entry point for the Human-in-the-Loop System.
Integrates all components and provides a complete production-ready system.
"""

import asyncio
import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import structlog
from pydantic_settings import BaseSettings
from pydantic import Field

from .models.workflow import Base
from .agents.orchestrator import SmolAgentOrchestrator
from .core.state_manager import StateManager
from .core.approval_engine import ApprovalEngine
from .core.resilience import ResilientExecutor, RetryConfig
from .observability.monitoring import ObservabilityManager
from .ui.approval_ui import ApprovalUIRenderer
from .integrations.channels import ChannelManager

# Configure structured logging
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


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost/human_in_loop",
        description="Database connection URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # HuggingFace
    hf_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    # Slack Integration
    slack_bot_token: Optional[str] = Field(default=None, description="Slack bot token")
    slack_signing_secret: Optional[str] = Field(default=None, description="Slack signing secret")
    slack_app_token: Optional[str] = Field(default=None, description="Slack app token")
    
    # Email Integration
    sendgrid_api_key: Optional[str] = Field(default=None, description="SendGrid API key")
    sendgrid_from_email: str = Field(default="noreply@company.com", description="From email address")
    
    # SMTP Configuration (alternative to SendGrid)
    smtp_host: Optional[str] = Field(default=None, description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", description="Secret key for sessions")
    encryption_key: str = Field(default="dev-encryption-key-32-chars!!", description="Encryption key")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_debug: bool = Field(default=False, description="Enable debug mode")
    
    # Workflow Configuration
    default_approval_timeout: int = Field(default=86400, description="Default approval timeout in seconds")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    
    # Monitoring
    prometheus_port: int = Field(default=8001, description="Prometheus metrics port")
    jaeger_endpoint: Optional[str] = Field(default=None, description="Jaeger tracing endpoint")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Frontend
    frontend_url: str = Field(default="http://localhost:3000", description="Frontend URL for CORS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ApplicationState:
    """Global application state container."""
    
    def __init__(self):
        self.settings = Settings()
        self.db_engine = None
        self.session_factory = None
        self.orchestrator = None
        self.observability = None
        self.resilient_executor = None
        
    async def initialize(self):
        """Initialize all application components."""
        logger.info("Initializing Human-in-the-Loop System", version="1.0.0")
        
        # Database setup
        self.db_engine = create_async_engine(
            self.settings.database_url,
            echo=self.settings.api_debug
        )
        
        self.session_factory = sessionmaker(
            self.db_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create database tables
        async with self.db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize core components with a session
        async with self.session_factory() as session:
            state_manager = StateManager(session)
            
            # Setup resilience
            self.resilient_executor = ResilientExecutor(
                state_manager,
                default_retry_config=RetryConfig(
                    max_attempts=self.settings.max_retry_attempts,
                    backoff_factor=self.settings.retry_backoff_factor
                )
            )
            
            # Setup observability
            self.observability = ObservabilityManager(
                state_manager,
                metrics_port=self.settings.prometheus_port,
                jaeger_endpoint=self.settings.jaeger_endpoint
            )
            
            # Setup orchestrator
            model_config = {
                "provider": "inference_client"
            }
            
            if self.settings.hf_token:
                os.environ["HF_TOKEN"] = self.settings.hf_token
            
            self.orchestrator = SmolAgentOrchestrator(
                db_session=session,
                model_config=model_config,
                agent_type="code"
            )
            
            # Configure integrations
            await self._configure_integrations()
        
        # Start monitoring
        await self.observability.start_monitoring()
        
        logger.info("Human-in-the-Loop System initialized successfully")
    
    async def _configure_integrations(self):
        """Configure communication channel integrations."""
        # Slack integration
        if self.settings.slack_bot_token and self.settings.slack_signing_secret:
            self.orchestrator.configure_slack_integration(
                bot_token=self.settings.slack_bot_token,
                signing_secret=self.settings.slack_signing_secret
            )
            logger.info("Slack integration configured")
        
        # Email integration
        if self.settings.sendgrid_api_key:
            self.orchestrator.configure_email_integration(
                sendgrid_api_key=self.settings.sendgrid_api_key
            )
            logger.info("SendGrid email integration configured")
        elif self.settings.smtp_host:
            smtp_config = {
                "host": self.settings.smtp_host,
                "port": self.settings.smtp_port,
                "use_tls": self.settings.smtp_use_tls,
                "from_email": self.settings.sendgrid_from_email
            }
            
            if self.settings.smtp_username:
                smtp_config.update({
                    "username": self.settings.smtp_username,
                    "password": self.settings.smtp_password
                })
            
            self.orchestrator.configure_email_integration(smtp_config=smtp_config)
            logger.info("SMTP email integration configured")
        
        # Webhook integration
        self.orchestrator.configure_webhook_integration()
        logger.info("Webhook integration configured")
    
    async def cleanup(self):
        """Cleanup application resources."""
        if self.observability:
            await self.observability.stop_monitoring()
        
        if self.db_engine:
            await self.db_engine.dispose()
        
        logger.info("Application cleanup completed")


# Global application state
app_state = ApplicationState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await app_state.initialize()
    yield
    # Shutdown
    await app_state.cleanup()


# FastAPI application
app = FastAPI(
    title="Human-in-the-Loop System",
    description="Event-driven stateful orchestration system with SmolAgents integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[app_state.settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db_session():
    """Dependency to get database session."""
    async with app_state.session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_orchestrator(session: AsyncSession = Depends(get_db_session)):
    """Dependency to get orchestrator with fresh session."""
    return SmolAgentOrchestrator(
        db_session=session,
        model_config={"provider": "inference_client"},
        agent_type="code"
    )


# API Routes

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Human-in-the-Loop System",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "SmolAgent Integration",
            "Multi-channel Approvals",
            "State Management & Rollback",
            "Event-driven Architecture",
            "Comprehensive Observability"
        ]
    }


@app.get("/health")
async def health_check():
    """System health check endpoint."""
    return app_state.observability.get_system_health()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    metrics_data = app_state.observability.metrics.get_metrics()
    return Response(content=metrics_data, media_type="text/plain")


@app.post("/workflows/")
async def create_workflow(
    workflow_data: Dict[str, Any],
    orchestrator: SmolAgentOrchestrator = Depends(get_orchestrator)
):
    """Create a new workflow."""
    try:
        workflow = await orchestrator.create_workflow(
            name=workflow_data.get("name", "API Workflow"),
            description=workflow_data.get("description", ""),
            workflow_definition=workflow_data.get("definition", {}),
            input_data=workflow_data.get("input_data", {}),
            created_by=workflow_data.get("created_by", "api_user")
        )
        
        return {
            "workflow_id": str(workflow.id),
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat()
        }
    
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    execution_data: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    orchestrator: SmolAgentOrchestrator = Depends(get_orchestrator)
):
    """Execute a workflow."""
    try:
        from uuid import UUID
        workflow_uuid = UUID(workflow_id)
        
        # Execute workflow in background
        background_tasks.add_task(
            orchestrator.execute_workflow,
            workflow_id=workflow_uuid,
            task=execution_data.get("task") if execution_data else None,
            context=execution_data.get("context", {}) if execution_data else {}
        )
        
        return {
            "message": "Workflow execution started",
            "workflow_id": workflow_id
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workflow ID format")
    except Exception as e:
        logger.error("Failed to execute workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    orchestrator: SmolAgentOrchestrator = Depends(get_orchestrator)
):
    """Get workflow status and progress."""
    try:
        from uuid import UUID
        workflow_uuid = UUID(workflow_id)
        
        status = await orchestrator.get_workflow_status(workflow_uuid)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return status
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workflow ID format")
    except Exception as e:
        logger.error("Failed to get workflow status", error=str(e), workflow_id=workflow_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/{workflow_id}/rollback")
async def rollback_workflow(
    workflow_id: str,
    rollback_data: Dict[str, Any],
    orchestrator: SmolAgentOrchestrator = Depends(get_orchestrator)
):
    """Rollback workflow to a checkpoint."""
    try:
        from uuid import UUID
        workflow_uuid = UUID(workflow_id)
        
        workflow = await orchestrator.rollback_workflow(
            checkpoint_name=rollback_data.get("checkpoint_name"),
            reason=rollback_data.get("reason", "Manual rollback via API")
        )
        
        return {
            "message": "Workflow rolled back successfully",
            "workflow_id": workflow_id,
            "rollback_point": workflow.rollback_point,
            "status": workflow.status.value
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workflow ID format")
    except Exception as e:
        logger.error("Failed to rollback workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/approvals/pending")
async def list_pending_approvals(
    requested_from: Optional[str] = None,
    workflow_id: Optional[str] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_db_session)
):
    """List pending approval requests."""
    try:
        approval_engine = ApprovalEngine(session, StateManager(session))
        
        workflow_uuid = None
        if workflow_id:
            from uuid import UUID
            workflow_uuid = UUID(workflow_id)
        
        approvals = await approval_engine.list_pending_approvals(
            requested_from=requested_from,
            workflow_id=workflow_uuid,
            limit=limit
        )
        
        return [
            {
                "id": str(approval.id),
                "workflow_id": str(approval.workflow_id),
                "title": approval.title,
                "description": approval.description,
                "approval_type": approval.approval_type,
                "requested_from": approval.requested_from,
                "created_at": approval.created_at.isoformat(),
                "expires_at": approval.expires_at.isoformat(),
                "priority": approval.priority
            }
            for approval in approvals
        ]
    
    except Exception as e:
        logger.error("Failed to list pending approvals", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approvals/{approval_id}/respond")
async def respond_to_approval(
    approval_id: str,
    response_data: Dict[str, Any],
    session: AsyncSession = Depends(get_db_session)
):
    """Respond to an approval request."""
    try:
        from uuid import UUID
        from ..models.workflow import ChannelType
        
        approval_uuid = UUID(approval_id)
        approval_engine = ApprovalEngine(session, StateManager(session))
        
        response = await approval_engine.respond_to_approval(
            approval_id=approval_uuid,
            approver_id=response_data.get("approver_id", "api_user"),
            decision=response_data["decision"],
            response_data=response_data.get("response_data", {}),
            feedback=response_data.get("feedback", ""),
            channel_type=ChannelType.UI
        )
        
        return {
            "message": "Approval response recorded successfully",
            "response_id": str(response.id),
            "decision": response.decision
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid approval ID format")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        logger.error("Failed to respond to approval", error=str(e), approval_id=approval_id)
        raise HTTPException(status_code=400, detail=str(e))


# Slack integration endpoint
@app.post("/slack/events")
async def slack_events(request: Dict[str, Any]):
    """Handle Slack events and interactions."""
    slack_app = app_state.orchestrator.get_slack_app()
    if slack_app:
        # This would integrate with the Slack app handler
        # Implementation depends on the specific Slack framework used
        return {"status": "received"}
    else:
        raise HTTPException(status_code=503, detail="Slack integration not configured")


if __name__ == "__main__":
    import uvicorn
    
    settings = Settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    )