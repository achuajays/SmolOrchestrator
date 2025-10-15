"""
Approval workflow engine with complete human-in-the-loop capabilities,
multi-channel notifications, and dynamic UI generation.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID, uuid4

import structlog
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.workflow import (
    ApprovalRequest, ApprovalResponse, ApprovalStatus, ChannelType,
    Workflow, WorkflowStatus, EventType
)
from .state_manager import StateManager

logger = structlog.get_logger(__name__)


class ApprovalEngine:
    """
    Core approval engine that handles human-in-the-loop workflows with
    multi-channel notifications and dynamic approval interfaces.
    """
    
    def __init__(self, db_session: AsyncSession, state_manager: StateManager):
        self.db = db_session
        self.state_manager = state_manager
        self.logger = logger.bind(component="approval_engine")
        
        # Registry of channel handlers
        self._channel_handlers = {}
        
        # Registry of approval response validators
        self._validators = {}
        
        # Registry of UI schema generators
        self._ui_generators = {}
    
    def register_channel_handler(self, channel_type: ChannelType, handler: Callable):
        """Register a handler for a specific communication channel."""
        self._channel_handlers[channel_type] = handler
        self.logger.info(f"Registered channel handler for {channel_type.value}")
    
    def register_validator(self, approval_type: str, validator: Callable):
        """Register a response validator for a specific approval type."""
        self._validators[approval_type] = validator
        self.logger.info(f"Registered validator for approval type {approval_type}")
    
    def register_ui_generator(self, approval_type: str, generator: Callable):
        """Register a UI schema generator for a specific approval type."""
        self._ui_generators[approval_type] = generator
        self.logger.info(f"Registered UI generator for approval type {approval_type}")
    
    async def request_approval(
        self,
        workflow_id: UUID,
        title: str,
        description: str,
        approval_type: str = "binary",
        requested_from: str = "admin",
        requested_by: str = "system",
        channels: Optional[List[ChannelType]] = None,
        ui_schema: Optional[Dict[str, Any]] = None,
        context_data: Optional[Dict[str, Any]] = None,
        timeout_hours: int = 24,
        required_approvers: Optional[List[str]] = None,
        approval_threshold: int = 1,
        priority: int = 0,
        **kwargs
    ) -> ApprovalRequest:
        """
        Create a new approval request and send notifications through configured channels.
        """
        # Set default channels
        if channels is None:
            channels = [ChannelType.UI, ChannelType.EMAIL]
        
        # Generate UI schema if not provided
        if ui_schema is None and approval_type in self._ui_generators:
            ui_schema = await self._ui_generators[approval_type](
                title, description, context_data or {}
            )
        
        # Create approval request
        approval = ApprovalRequest(
            id=uuid4(),
            workflow_id=workflow_id,
            title=title,
            description=description,
            approval_type=approval_type,
            requested_from=requested_from,
            requested_by=requested_by,
            channels=[ch.value for ch in channels],
            ui_schema=ui_schema,
            context_data=context_data or {},
            expires_at=datetime.now(timezone.utc) + timedelta(hours=timeout_hours),
            required_approvers=required_approvers or [requested_from],
            approval_threshold=approval_threshold,
            priority=priority,
            **kwargs
        )
        
        self.db.add(approval)
        await self.db.flush()
        
        # Update workflow status
        await self.state_manager.transition_workflow_status(
            workflow_id,
            WorkflowStatus.WAITING_APPROVAL,
            triggered_by=requested_by,
            reason=f"Approval requested: {title}"
        )
        
        # Log approval request event
        await self.state_manager._create_event(
            workflow_id,
            EventType.APPROVAL_REQUESTED,
            f"Approval requested: {title}",
            event_data={
                "approval_id": str(approval.id),
                "approval_type": approval_type,
                "requested_from": requested_from,
                "channels": [ch.value for ch in channels],
                "timeout_hours": timeout_hours
            },
            triggered_by=requested_by
        )
        
        # Send notifications through all configured channels
        for channel in channels:
            if channel in self._channel_handlers:
                try:
                    await self._channel_handlers[channel](approval)
                    self.logger.info(
                        "Approval notification sent",
                        approval_id=str(approval.id),
                        channel=channel.value
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to send approval notification",
                        approval_id=str(approval.id),
                        channel=channel.value,
                        error=str(e)
                    )
            else:
                self.logger.warning(
                    "No handler registered for channel",
                    channel=channel.value,
                    approval_id=str(approval.id)
                )
        
        await self.db.commit()
        self.logger.info(
            "Approval request created",
            approval_id=str(approval.id),
            workflow_id=str(workflow_id),
            title=title
        )
        
        return approval
    
    async def respond_to_approval(
        self,
        approval_id: UUID,
        approver_id: str,
        decision: str,  # "approved", "rejected"
        response_data: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        channel_type: ChannelType = ChannelType.UI,
        channel_metadata: Optional[Dict[str, Any]] = None
    ) -> ApprovalResponse:
        """
        Process an approval response and update workflow state accordingly.
        """
        # Get the approval request
        approval = await self.get_approval(approval_id)
        if not approval:
            raise ValueError(f"Approval request {approval_id} not found")
        
        if approval.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Approval request {approval_id} is not pending (status: {approval.status})"
            )
        
        # Check if approval has expired
        if datetime.now(timezone.utc) > approval.expires_at:
            await self._expire_approval(approval)
            raise ValueError(f"Approval request {approval_id} has expired")
        
        # Validate the response if validator exists
        if approval.approval_type in self._validators:
            try:
                is_valid, error_msg = await self._validators[approval.approval_type](
                    response_data, approval.ui_schema, approval.context_data
                )
                if not is_valid:
                    raise ValueError(f"Invalid approval response: {error_msg}")
            except Exception as e:
                self.logger.error(
                    "Validation failed for approval response",
                    approval_id=str(approval_id),
                    error=str(e)
                )
                raise ValueError(f"Response validation failed: {str(e)}")
        
        # Create approval response record
        response = ApprovalResponse(
            id=uuid4(),
            request_id=approval_id,
            approver_id=approver_id,
            decision=decision,
            response_data=response_data or {},
            feedback=feedback,
            channel_type=channel_type,
            channel_metadata=channel_metadata or {}
        )
        
        self.db.add(response)
        await self.db.flush()
        
        # Update approval request
        approval.responded_at = datetime.now(timezone.utc)
        
        # Initialize approved_by list if it doesn't exist
        if not approval.approved_by:
            approval.approved_by = []
        
        # Add approver to the list if not already present
        if approver_id not in approval.approved_by:
            approval.approved_by.append(approver_id)
        
        # Check if approval threshold is met
        approved_count = sum(
            1 for resp in approval.responses 
            if resp.decision == "approved"
        )
        
        if decision == "approved":
            approved_count += 1
        
        # Determine final approval status
        if decision == "rejected":
            approval.status = ApprovalStatus.REJECTED
            approval.decision = "rejected"
            workflow_status = WorkflowStatus.REJECTED
        elif approved_count >= approval.approval_threshold:
            approval.status = ApprovalStatus.APPROVED
            approval.decision = "approved"
            workflow_status = WorkflowStatus.APPROVED
        else:
            # Still need more approvals
            await self.db.commit()
            self.logger.info(
                "Partial approval received",
                approval_id=str(approval_id),
                approver=approver_id,
                approved_count=approved_count,
                threshold=approval.approval_threshold
            )
            return response
        
        # Update approval response data
        approval.response_data = response_data
        
        # Update workflow status
        await self.state_manager.transition_workflow_status(
            approval.workflow_id,
            workflow_status,
            triggered_by=approver_id,
            reason=f"Approval {decision} by {approver_id}"
        )
        
        # Log approval response event
        await self.state_manager._create_event(
            approval.workflow_id,
            EventType.APPROVAL_RESPONDED,
            f"Approval {decision}: {approval.title}",
            event_data={
                "approval_id": str(approval_id),
                "approver_id": approver_id,
                "decision": decision,
                "response_data": response_data,
                "channel": channel_type.value
            },
            triggered_by=approver_id
        )
        
        await self.db.commit()
        self.logger.info(
            "Approval response processed",
            approval_id=str(approval_id),
            approver=approver_id,
            decision=decision,
            workflow_status=workflow_status.value
        )
        
        return response
    
    async def get_approval(
        self,
        approval_id: UUID,
        include_responses: bool = True
    ) -> Optional[ApprovalRequest]:
        """Get approval request with optional response data."""
        stmt = select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
        
        if include_responses:
            stmt = stmt.options(selectinload(ApprovalRequest.responses))
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_pending_approvals(
        self,
        requested_from: Optional[str] = None,
        workflow_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[ApprovalRequest]:
        """List pending approval requests."""
        stmt = select(ApprovalRequest).where(
            ApprovalRequest.status == ApprovalStatus.PENDING
        )
        
        if requested_from:
            stmt = stmt.where(ApprovalRequest.requested_from == requested_from)
        if workflow_id:
            stmt = stmt.where(ApprovalRequest.workflow_id == workflow_id)
        
        stmt = stmt.order_by(ApprovalRequest.priority.desc(), ApprovalRequest.created_at)
        stmt = stmt.limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def expire_approval(self, approval_id: UUID) -> bool:
        """Manually expire an approval request."""
        approval = await self.get_approval(approval_id)
        if not approval:
            return False
        
        return await self._expire_approval(approval)
    
    async def _expire_approval(self, approval: ApprovalRequest) -> bool:
        """Internal method to expire an approval request."""
        if approval.status != ApprovalStatus.PENDING:
            return False
        
        approval.status = ApprovalStatus.EXPIRED
        approval.responded_at = datetime.now(timezone.utc)
        
        # Update workflow status to failed due to timeout
        await self.state_manager.transition_workflow_status(
            approval.workflow_id,
            WorkflowStatus.FAILED,
            triggered_by="system",
            reason=f"Approval expired: {approval.title}"
        )
        
        # Log expiration event
        await self.state_manager._create_event(
            approval.workflow_id,
            EventType.APPROVAL_EXPIRED,
            f"Approval expired: {approval.title}",
            event_data={
                "approval_id": str(approval.id),
                "expired_at": datetime.now(timezone.utc).isoformat()
            },
            triggered_by="system"
        )
        
        await self.db.commit()
        self.logger.info(
            "Approval request expired",
            approval_id=str(approval.id),
            title=approval.title
        )
        
        return True
    
    async def cancel_approval(
        self,
        approval_id: UUID,
        cancelled_by: str = "system",
        reason: Optional[str] = None
    ) -> bool:
        """Cancel a pending approval request."""
        approval = await self.get_approval(approval_id)
        if not approval or approval.status != ApprovalStatus.PENDING:
            return False
        
        approval.status = ApprovalStatus.CANCELLED
        approval.responded_at = datetime.now(timezone.utc)
        
        # Update workflow status
        await self.state_manager.transition_workflow_status(
            approval.workflow_id,
            WorkflowStatus.CANCELLED,
            triggered_by=cancelled_by,
            reason=reason or f"Approval cancelled: {approval.title}"
        )
        
        # Log cancellation event
        await self.state_manager._create_event(
            approval.workflow_id,
            EventType.APPROVAL_RESPONDED,
            f"Approval cancelled: {approval.title}",
            event_data={
                "approval_id": str(approval.id),
                "cancelled_by": cancelled_by,
                "reason": reason
            },
            triggered_by=cancelled_by
        )
        
        await self.db.commit()
        self.logger.info(
            "Approval request cancelled",
            approval_id=str(approval.id),
            cancelled_by=cancelled_by,
            reason=reason
        )
        
        return True
    
    async def check_expired_approvals(self) -> List[UUID]:
        """Check for and expire any approvals that have passed their timeout."""
        current_time = datetime.now(timezone.utc)
        
        stmt = select(ApprovalRequest).where(
            and_(
                ApprovalRequest.status == ApprovalStatus.PENDING,
                ApprovalRequest.expires_at < current_time
            )
        )
        
        result = await self.db.execute(stmt)
        expired_approvals = result.scalars().all()
        
        expired_ids = []
        for approval in expired_approvals:
            await self._expire_approval(approval)
            expired_ids.append(approval.id)
        
        if expired_ids:
            self.logger.info(f"Expired {len(expired_ids)} approval requests")
        
        return expired_ids


class ApprovalUISchemaGenerator:
    """
    Generates dynamic UI schemas for different approval types.
    """
    
    @staticmethod
    async def binary_approval(title: str, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI schema for binary (approve/reject) approval."""
        return {
            "type": "object",
            "title": title,
            "description": description,
            "properties": {
                "decision": {
                    "type": "string",
                    "title": "Decision",
                    "enum": ["approved", "rejected"],
                    "enumNames": ["Approve", "Reject"],
                    "default": "approved"
                },
                "feedback": {
                    "type": "string",
                    "title": "Feedback (Optional)",
                    "description": "Additional comments or feedback",
                    "format": "textarea"
                }
            },
            "required": ["decision"],
            "ui": {
                "decision": {
                    "ui:widget": "radio",
                    "ui:options": {
                        "inline": True
                    }
                },
                "feedback": {
                    "ui:widget": "textarea",
                    "ui:options": {
                        "rows": 3
                    }
                }
            }
        }
    
    @staticmethod
    async def form_approval(title: str, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI schema for form-based approval with custom fields."""
        fields = context.get("form_fields", [])
        
        properties = {
            "decision": {
                "type": "string",
                "title": "Decision",
                "enum": ["approved", "rejected", "modified"],
                "enumNames": ["Approve", "Reject", "Request Changes"],
                "default": "approved"
            }
        }
        
        ui_schema = {
            "decision": {
                "ui:widget": "radio"
            }
        }
        
        # Add custom fields from context
        for field in fields:
            field_name = field.get("name")
            field_type = field.get("type", "string")
            field_title = field.get("title", field_name.title())
            field_required = field.get("required", False)
            
            properties[field_name] = {
                "type": field_type,
                "title": field_title
            }
            
            if field.get("description"):
                properties[field_name]["description"] = field["description"]
            
            if field.get("enum"):
                properties[field_name]["enum"] = field["enum"]
            
            # Add UI configuration
            ui_config = field.get("ui", {})
            if ui_config:
                ui_schema[field_name] = ui_config
        
        properties["feedback"] = {
            "type": "string",
            "title": "Additional Comments",
            "format": "textarea"
        }
        
        ui_schema["feedback"] = {
            "ui:widget": "textarea",
            "ui:options": {"rows": 3}
        }
        
        required_fields = ["decision"]
        required_fields.extend([
            field["name"] for field in fields 
            if field.get("required", False)
        ])
        
        return {
            "type": "object",
            "title": title,
            "description": description,
            "properties": properties,
            "required": required_fields,
            "ui": ui_schema
        }
    
    @staticmethod
    async def budget_approval(title: str, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI schema for budget/financial approval."""
        amount = context.get("amount", 0)
        currency = context.get("currency", "USD")
        
        return {
            "type": "object",
            "title": title,
            "description": description,
            "properties": {
                "decision": {
                    "type": "string",
                    "title": "Decision",
                    "enum": ["approved", "rejected", "modified"],
                    "enumNames": ["Approve", "Reject", "Request Changes"]
                },
                "approved_amount": {
                    "type": "number",
                    "title": f"Approved Amount ({currency})",
                    "default": amount,
                    "minimum": 0
                },
                "budget_category": {
                    "type": "string",
                    "title": "Budget Category",
                    "enum": ["operations", "marketing", "development", "other"],
                    "enumNames": ["Operations", "Marketing", "Development", "Other"]
                },
                "justification": {
                    "type": "string",
                    "title": "Justification",
                    "description": "Explain the reasoning for this decision",
                    "format": "textarea"
                },
                "conditions": {
                    "type": "string",
                    "title": "Conditions/Requirements",
                    "description": "Any conditions or requirements for this approval",
                    "format": "textarea"
                }
            },
            "required": ["decision", "justification"],
            "ui": {
                "decision": {"ui:widget": "radio"},
                "approved_amount": {"ui:widget": "updown"},
                "justification": {
                    "ui:widget": "textarea",
                    "ui:options": {"rows": 3}
                },
                "conditions": {
                    "ui:widget": "textarea",
                    "ui:options": {"rows": 2}
                }
            }
        }


class ApprovalValidator:
    """
    Validates approval responses based on type and schema.
    """
    
    @staticmethod
    async def binary_validator(
        response_data: Dict[str, Any],
        ui_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate binary approval response."""
        if not response_data:
            return False, "Response data is required"
        
        decision = response_data.get("decision")
        if not decision or decision not in ["approved", "rejected"]:
            return False, "Decision must be 'approved' or 'rejected'"
        
        return True, None
    
    @staticmethod
    async def form_validator(
        response_data: Dict[str, Any],
        ui_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate form-based approval response."""
        if not response_data:
            return False, "Response data is required"
        
        required_fields = ui_schema.get("required", [])
        
        for field in required_fields:
            if field not in response_data or not response_data[field]:
                return False, f"Required field '{field}' is missing or empty"
        
        decision = response_data.get("decision")
        if decision not in ["approved", "rejected", "modified"]:
            return False, "Decision must be 'approved', 'rejected', or 'modified'"
        
        return True, None
    
    @staticmethod
    async def budget_validator(
        response_data: Dict[str, Any],
        ui_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate budget approval response."""
        if not response_data:
            return False, "Response data is required"
        
        decision = response_data.get("decision")
        if decision not in ["approved", "rejected", "modified"]:
            return False, "Decision must be 'approved', 'rejected', or 'modified'"
        
        if decision in ["approved", "modified"]:
            approved_amount = response_data.get("approved_amount")
            if approved_amount is None or approved_amount < 0:
                return False, "Approved amount must be a non-negative number"
            
            original_amount = context.get("amount", 0)
            max_amount = context.get("max_amount", original_amount * 2)
            
            if approved_amount > max_amount:
                return False, f"Approved amount cannot exceed {max_amount}"
        
        justification = response_data.get("justification")
        if not justification or len(justification.strip()) < 10:
            return False, "Justification must be at least 10 characters"
        
        return True, None