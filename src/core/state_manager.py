"""
Core state management service with complete rollback and recovery capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

import structlog
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.workflow import (
    Workflow, WorkflowStatus, WorkflowEvent, WorkflowCheckpoint, 
    EventType, ApprovalRequest, ApprovalStatus
)

logger = structlog.get_logger(__name__)


class StateManager:
    """
    Core state management service with complete workflow lifecycle management,
    rollback capabilities, and event-driven state transitions.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.logger = logger.bind(component="state_manager")
    
    async def create_workflow(
        self,
        name: str,
        definition: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
        **kwargs
    ) -> Workflow:
        """Create a new workflow with initial state."""
        workflow = Workflow(
            id=uuid4(),
            name=name,
            definition=definition,
            input_data=input_data or {},
            current_state={
                "step_index": 0,
                "step_name": "start",
                "variables": input_data or {},
                "execution_context": {},
                "agent_state": {}
            },
            created_by=created_by,
            status=WorkflowStatus.PENDING,
            total_steps=len(definition.get("steps", [])),
            **kwargs
        )
        
        self.db.add(workflow)
        await self.db.flush()
        
        # Log workflow creation event
        await self._create_event(
            workflow.id,
            EventType.WORKFLOW_CREATED,
            "Workflow created",
            event_data={"definition": definition, "input_data": input_data},
            new_state=workflow.current_state,
            triggered_by=created_by
        )
        
        # Create initial checkpoint
        await self.create_checkpoint(
            workflow.id,
            "initial_state",
            "start",
            "Initial workflow state",
            created_by=created_by
        )
        
        await self.db.commit()
        self.logger.info("Workflow created", workflow_id=str(workflow.id), name=name)
        return workflow
    
    async def update_workflow_state(
        self,
        workflow_id: UUID,
        new_state: Dict[str, Any],
        step_name: Optional[str] = None,
        triggered_by: str = "system"
    ) -> Workflow:
        """Update workflow state with automatic checkpoint creation."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        previous_state = workflow.current_state.copy() if workflow.current_state else {}
        
        # Update workflow state
        workflow.current_state = new_state
        if step_name:
            workflow.current_step = step_name
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Log state change event
        await self._create_event(
            workflow_id,
            EventType.STATE_CHECKPOINT,
            f"State updated for step: {step_name or 'unknown'}",
            event_data={"step_name": step_name},
            previous_state=previous_state,
            new_state=new_state,
            step_name=step_name,
            triggered_by=triggered_by
        )
        
        await self.db.commit()
        self.logger.info("Workflow state updated", workflow_id=str(workflow_id), step=step_name)
        return workflow
    
    async def transition_workflow_status(
        self,
        workflow_id: UUID,
        new_status: WorkflowStatus,
        triggered_by: str = "system",
        reason: Optional[str] = None
    ) -> Workflow:
        """Transition workflow to a new status with proper event logging."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        previous_status = workflow.status
        workflow.status = new_status
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Set completion timestamp if workflow is completed
        if new_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, 
                         WorkflowStatus.CANCELLED, WorkflowStatus.ROLLED_BACK]:
            workflow.completed_at = datetime.now(timezone.utc)
        
        # Update started timestamp if starting
        if new_status == WorkflowStatus.RUNNING and not workflow.started_at:
            workflow.started_at = datetime.now(timezone.utc)
        
        # Log appropriate event
        event_type_mapping = {
            WorkflowStatus.RUNNING: EventType.WORKFLOW_STARTED,
            WorkflowStatus.WAITING_APPROVAL: EventType.WORKFLOW_PAUSED,
            WorkflowStatus.COMPLETED: EventType.WORKFLOW_COMPLETED,
            WorkflowStatus.FAILED: EventType.WORKFLOW_FAILED,
            WorkflowStatus.CANCELLED: EventType.WORKFLOW_CANCELLED,
            WorkflowStatus.ROLLED_BACK: EventType.ROLLBACK_COMPLETED,
        }
        
        event_type = event_type_mapping.get(new_status, EventType.WORKFLOW_STARTED)
        
        await self._create_event(
            workflow_id,
            event_type,
            f"Workflow status changed from {previous_status.value} to {new_status.value}",
            event_data={
                "previous_status": previous_status.value,
                "new_status": new_status.value,
                "reason": reason
            },
            triggered_by=triggered_by
        )
        
        await self.db.commit()
        self.logger.info(
            "Workflow status transitioned",
            workflow_id=str(workflow_id),
            from_status=previous_status.value,
            to_status=new_status.value,
            reason=reason
        )
        return workflow
    
    async def create_checkpoint(
        self,
        workflow_id: UUID,
        checkpoint_name: str,
        step_name: str,
        description: Optional[str] = None,
        is_automatic: bool = True,
        created_by: str = "system"
    ) -> WorkflowCheckpoint:
        """Create a state checkpoint for rollback purposes."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            checkpoint_name=checkpoint_name,
            step_name=step_name,
            description=description or f"Checkpoint for step {step_name}",
            state_snapshot=workflow.current_state,
            execution_context={
                "status": workflow.status.value,
                "current_step": workflow.current_step,
                "step_index": workflow.current_state.get("step_index", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            is_automatic=is_automatic,
            created_by=created_by
        )
        
        self.db.add(checkpoint)
        await self.db.flush()
        
        # Update workflow's checkpoint reference
        workflow.checkpoint_data = checkpoint.state_snapshot
        workflow.rollback_point = checkpoint_name
        
        await self._create_event(
            workflow_id,
            EventType.STATE_CHECKPOINT,
            f"Checkpoint created: {checkpoint_name}",
            event_data={"checkpoint_name": checkpoint_name, "step_name": step_name},
            triggered_by=created_by
        )
        
        await self.db.commit()
        self.logger.info(
            "Checkpoint created",
            workflow_id=str(workflow_id),
            checkpoint=checkpoint_name,
            step=step_name
        )
        return checkpoint
    
    async def rollback_workflow(
        self,
        workflow_id: UUID,
        checkpoint_name: Optional[str] = None,
        triggered_by: str = "system",
        reason: Optional[str] = None
    ) -> Workflow:
        """Rollback workflow to a specific checkpoint or the latest one."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Find the target checkpoint
        if checkpoint_name:
            stmt = select(WorkflowCheckpoint).where(
                and_(
                    WorkflowCheckpoint.workflow_id == workflow_id,
                    WorkflowCheckpoint.checkpoint_name == checkpoint_name
                )
            )
        else:
            # Get the latest checkpoint
            stmt = select(WorkflowCheckpoint).where(
                WorkflowCheckpoint.workflow_id == workflow_id
            ).order_by(desc(WorkflowCheckpoint.created_at))
        
        result = await self.db.execute(stmt)
        checkpoint = result.scalar_one_or_none()
        
        if not checkpoint:
            raise ValueError(
                f"No checkpoint found for workflow {workflow_id}"
                f"{f' with name {checkpoint_name}' if checkpoint_name else ''}"
            )
        
        # Store current state for rollback event
        previous_state = workflow.current_state.copy() if workflow.current_state else {}
        
        # Restore workflow state from checkpoint
        workflow.current_state = checkpoint.state_snapshot
        workflow.current_step = checkpoint.step_name
        workflow.status = WorkflowStatus.ROLLED_BACK
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Reset any pending approvals
        pending_approvals = await self.db.execute(
            select(ApprovalRequest).where(
                and_(
                    ApprovalRequest.workflow_id == workflow_id,
                    ApprovalRequest.status == ApprovalStatus.PENDING
                )
            )
        )
        
        for approval in pending_approvals.scalars():
            approval.status = ApprovalStatus.CANCELLED
        
        # Log rollback events
        await self._create_event(
            workflow_id,
            EventType.ROLLBACK_INITIATED,
            f"Rollback initiated to checkpoint: {checkpoint.checkpoint_name}",
            event_data={
                "checkpoint_name": checkpoint.checkpoint_name,
                "target_step": checkpoint.step_name,
                "reason": reason
            },
            previous_state=previous_state,
            new_state=checkpoint.state_snapshot,
            triggered_by=triggered_by
        )
        
        await self._create_event(
            workflow_id,
            EventType.ROLLBACK_COMPLETED,
            f"Rollback completed to checkpoint: {checkpoint.checkpoint_name}",
            event_data={
                "checkpoint_name": checkpoint.checkpoint_name,
                "restored_step": checkpoint.step_name
            },
            triggered_by=triggered_by
        )
        
        await self.db.commit()
        self.logger.info(
            "Workflow rolled back",
            workflow_id=str(workflow_id),
            checkpoint=checkpoint.checkpoint_name,
            target_step=checkpoint.step_name,
            reason=reason
        )
        return workflow
    
    async def get_workflow(
        self,
        workflow_id: UUID,
        include_events: bool = False,
        include_approvals: bool = False,
        include_checkpoints: bool = False
    ) -> Optional[Workflow]:
        """Get workflow with optional related data."""
        stmt = select(Workflow).where(Workflow.id == workflow_id)
        
        if include_events:
            stmt = stmt.options(selectinload(Workflow.events))
        if include_approvals:
            stmt = stmt.options(selectinload(Workflow.approvals))
        if include_checkpoints:
            stmt = stmt.options(selectinload(Workflow.checkpoints))
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Workflow]:
        """List workflows with optional filtering."""
        stmt = select(Workflow)
        
        if status:
            stmt = stmt.where(Workflow.status == status)
        if created_by:
            stmt = stmt.where(Workflow.created_by == created_by)
        
        stmt = stmt.offset(offset).limit(limit).order_by(desc(Workflow.created_at))
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_workflow_events(
        self,
        workflow_id: UUID,
        event_types: Optional[List[EventType]] = None,
        limit: int = 100
    ) -> List[WorkflowEvent]:
        """Get workflow events with optional filtering by type."""
        stmt = select(WorkflowEvent).where(WorkflowEvent.workflow_id == workflow_id)
        
        if event_types:
            stmt = stmt.where(WorkflowEvent.event_type.in_(event_types))
        
        stmt = stmt.order_by(desc(WorkflowEvent.created_at)).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_workflow_checkpoints(
        self,
        workflow_id: UUID,
        limit: int = 10
    ) -> List[WorkflowCheckpoint]:
        """Get workflow checkpoints ordered by creation time."""
        stmt = select(WorkflowCheckpoint).where(
            WorkflowCheckpoint.workflow_id == workflow_id
        ).order_by(desc(WorkflowCheckpoint.created_at)).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def cleanup_expired_workflows(self, hours: int = 24) -> int:
        """Clean up expired workflows and their related data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Find expired workflows
        stmt = select(Workflow).where(
            and_(
                Workflow.expires_at < cutoff_time,
                Workflow.status.in_([
                    WorkflowStatus.PENDING,
                    WorkflowStatus.WAITING_APPROVAL
                ])
            )
        )
        
        result = await self.db.execute(stmt)
        expired_workflows = result.scalars().all()
        
        count = 0
        for workflow in expired_workflows:
            await self.transition_workflow_status(
                workflow.id,
                WorkflowStatus.CANCELLED,
                triggered_by="system",
                reason="Workflow expired"
            )
            count += 1
        
        self.logger.info(f"Cleaned up {count} expired workflows")
        return count
    
    async def _create_event(
        self,
        workflow_id: UUID,
        event_type: EventType,
        description: str,
        event_data: Optional[Dict[str, Any]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        step_name: Optional[str] = None,
        triggered_by: str = "system"
    ) -> WorkflowEvent:
        """Create a workflow event for audit trail."""
        event = WorkflowEvent(
            workflow_id=workflow_id,
            event_type=event_type,
            event_name=event_type.value.replace("_", " ").title(),
            description=description,
            event_data=event_data or {},
            previous_state=previous_state,
            new_state=new_state,
            step_name=step_name,
            triggered_by=triggered_by
        )
        
        self.db.add(event)
        await self.db.flush()
        return event


class WorkflowRecoveryService:
    """Service for handling workflow recovery and failure scenarios."""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.logger = logger.bind(component="recovery_service")
    
    async def recover_failed_workflows(self) -> List[UUID]:
        """Attempt to recover workflows that failed due to system errors."""
        failed_workflows = await self.state_manager.list_workflows(
            status=WorkflowStatus.FAILED
        )
        
        recovered_ids = []
        for workflow in failed_workflows:
            try:
                # Check if the failure was recoverable
                if self._is_recoverable_failure(workflow):
                    # Rollback to last checkpoint and retry
                    await self.state_manager.rollback_workflow(
                        workflow.id,
                        triggered_by="recovery_service",
                        reason="Automatic recovery attempt"
                    )
                    
                    # Reset to running status for retry
                    await self.state_manager.transition_workflow_status(
                        workflow.id,
                        WorkflowStatus.RUNNING,
                        triggered_by="recovery_service",
                        reason="Recovery attempt"
                    )
                    
                    recovered_ids.append(workflow.id)
                    self.logger.info("Workflow recovered", workflow_id=str(workflow.id))
            
            except Exception as e:
                self.logger.error(
                    "Failed to recover workflow",
                    workflow_id=str(workflow.id),
                    error=str(e)
                )
        
        return recovered_ids
    
    def _is_recoverable_failure(self, workflow: Workflow) -> bool:
        """Determine if a workflow failure is recoverable."""
        # Check retry count
        if workflow.retry_count >= workflow.max_retries:
            return False
        
        # Check failure reason (if available)
        if workflow.failure_reason:
            recoverable_reasons = [
                "timeout", "network_error", "temporary_failure",
                "service_unavailable", "rate_limit"
            ]
            return any(reason in workflow.failure_reason.lower() 
                      for reason in recoverable_reasons)
        
        return True  # Assume recoverable if no specific reason