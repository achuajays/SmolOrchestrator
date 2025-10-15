"""
Workflow state management models with complete state tracking and rollback capabilities.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, 
    JSON, String, Text, func
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Index

Base = declarative_base()


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class ApprovalStatus(str, Enum):
    """Individual approval request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ChannelType(str, Enum):
    """Communication channel types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    UI = "ui"


class EventType(str, Enum):
    """Workflow event types for comprehensive audit trail."""
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESPONDED = "approval_responded"
    APPROVAL_EXPIRED = "approval_expired"
    STATE_CHECKPOINT = "state_checkpoint"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_COMPLETED = "rollback_completed"
    RETRY_ATTEMPTED = "retry_attempted"


class Workflow(Base):
    """Main workflow entity with complete state management."""
    
    __tablename__ = "workflows"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Status and execution state
    status = Column(SQLEnum(WorkflowStatus), default=WorkflowStatus.PENDING, nullable=False)
    current_step = Column(String(255))
    total_steps = Column(Integer)
    
    # Workflow configuration
    definition = Column(JSON, nullable=False)  # Workflow steps and configuration
    input_data = Column(JSON)  # Initial input data
    output_data = Column(JSON)  # Final output data
    
    # State management
    current_state = Column(JSON)  # Current execution state
    checkpoint_data = Column(JSON)  # Latest checkpoint for rollback
    rollback_point = Column(String(255))  # Step to rollback to
    
    # Metadata
    created_by = Column(String(255))
    assigned_to = Column(String(255))  # Current approver/assignee
    priority = Column(Integer, default=0)
    tags = Column(JSON)  # Flexible tagging system
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Retry and failure handling
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    failure_reason = Column(Text)
    
    # Relationships
    approvals = relationship("ApprovalRequest", back_populates="workflow", cascade="all, delete-orphan")
    events = relationship("WorkflowEvent", back_populates="workflow", cascade="all, delete-orphan")
    checkpoints = relationship("WorkflowCheckpoint", back_populates="workflow", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_workflow_status", "status"),
        Index("idx_workflow_created_by", "created_by"),
        Index("idx_workflow_assigned_to", "assigned_to"),
        Index("idx_workflow_created_at", "created_at"),
    )


class ApprovalRequest(Base):
    """Individual approval requests with multi-channel support."""
    
    __tablename__ = "approval_requests"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(PGUUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False)
    
    # Approval details
    title = Column(String(255), nullable=False)
    description = Column(Text)
    approval_type = Column(String(50))  # "binary", "text", "form", "custom"
    
    # Status and response
    status = Column(SQLEnum(ApprovalStatus), default=ApprovalStatus.PENDING, nullable=False)
    response_data = Column(JSON)  # Approval response/feedback
    decision = Column(String(50))  # "approved", "rejected", "modified"
    
    # Configuration
    ui_schema = Column(JSON)  # Dynamic UI configuration
    validation_schema = Column(JSON)  # Response validation rules
    required_approvers = Column(JSON)  # List of required approvers
    approval_threshold = Column(Integer, default=1)  # Minimum approvals needed
    
    # Channel configuration
    channels = Column(JSON)  # List of notification channels
    channel_configs = Column(JSON)  # Channel-specific configurations
    
    # Assignee information
    requested_from = Column(String(255))  # Primary approver
    requested_by = Column(String(255))  # Requester
    approved_by = Column(JSON)  # List of approvers who responded
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    responded_at = Column(DateTime(timezone=True))
    
    # Metadata
    priority = Column(Integer, default=0)
    tags = Column(JSON)
    context_data = Column(JSON)  # Additional context for approval
    
    # Relationships
    workflow = relationship("Workflow", back_populates="approvals")
    responses = relationship("ApprovalResponse", back_populates="request", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_approval_workflow_id", "workflow_id"),
        Index("idx_approval_status", "status"),
        Index("idx_approval_requested_from", "requested_from"),
        Index("idx_approval_expires_at", "expires_at"),
    )


class ApprovalResponse(Base):
    """Individual responses to approval requests (for multi-approver scenarios)."""
    
    __tablename__ = "approval_responses"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id = Column(PGUUID(as_uuid=True), ForeignKey("approval_requests.id"), nullable=False)
    
    # Response details
    approver_id = Column(String(255), nullable=False)
    decision = Column(String(50), nullable=False)  # "approved", "rejected"
    response_data = Column(JSON)  # Structured response data
    feedback = Column(Text)  # Free-text feedback
    
    # Channel information
    channel_type = Column(SQLEnum(ChannelType), nullable=False)
    channel_metadata = Column(JSON)  # Channel-specific metadata
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    request = relationship("ApprovalRequest", back_populates="responses")
    
    # Indexes
    __table_args__ = (
        Index("idx_response_request_id", "request_id"),
        Index("idx_response_approver_id", "approver_id"),
    )


class WorkflowEvent(Base):
    """Comprehensive audit trail for all workflow events."""
    
    __tablename__ = "workflow_events"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(PGUUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False)
    
    # Event details
    event_type = Column(SQLEnum(EventType), nullable=False)
    event_name = Column(String(255))  # Human-readable event name
    description = Column(Text)
    
    # Event data
    event_data = Column(JSON)  # Structured event data
    previous_state = Column(JSON)  # State before event
    new_state = Column(JSON)  # State after event
    
    # Context
    triggered_by = Column(String(255))  # User/system that triggered event
    step_name = Column(String(255))  # Workflow step if applicable
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    workflow = relationship("Workflow", back_populates="events")
    
    # Indexes
    __table_args__ = (
        Index("idx_event_workflow_id", "workflow_id"),
        Index("idx_event_type", "event_type"),
        Index("idx_event_created_at", "created_at"),
    )


class WorkflowCheckpoint(Base):
    """State checkpoints for rollback capabilities."""
    
    __tablename__ = "workflow_checkpoints"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(PGUUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False)
    
    # Checkpoint details
    checkpoint_name = Column(String(255), nullable=False)
    step_name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # State data
    state_snapshot = Column(JSON, nullable=False)  # Complete state at checkpoint
    execution_context = Column(JSON)  # Execution context for restoration
    
    # Metadata
    is_automatic = Column(Boolean, default=True)
    created_by = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    workflow = relationship("Workflow", back_populates="checkpoints")
    
    # Indexes
    __table_args__ = (
        Index("idx_checkpoint_workflow_id", "workflow_id"),
        Index("idx_checkpoint_name", "checkpoint_name"),
        Index("idx_checkpoint_created_at", "created_at"),
    )


class ChannelConfiguration(Base):
    """Configuration for different communication channels."""
    
    __tablename__ = "channel_configurations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Channel details
    name = Column(String(255), nullable=False, unique=True)
    channel_type = Column(SQLEnum(ChannelType), nullable=False)
    description = Column(Text)
    
    # Configuration
    config_data = Column(JSON, nullable=False)  # Channel-specific config
    default_template = Column(JSON)  # Default message templates
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_channel_type", "channel_type"),
        Index("idx_channel_active", "is_active"),
    )


class WorkflowTemplate(Base):
    """Reusable workflow templates."""
    
    __tablename__ = "workflow_templates"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Template details
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    category = Column(String(100))
    
    # Template configuration
    definition = Column(JSON, nullable=False)  # Workflow definition template
    default_config = Column(JSON)  # Default configuration values
    required_params = Column(JSON)  # Required parameters for instantiation
    
    # Metadata
    created_by = Column(String(255))
    version = Column(String(50), default="1.0.0")
    tags = Column(JSON)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_template_category", "category"),
        Index("idx_template_active", "is_active"),
    )