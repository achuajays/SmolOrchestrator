"""
SmolAgent orchestrator that integrates human-in-the-loop approvals
with SmolAgents framework for complete workflow orchestration.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import UUID, uuid4

import structlog
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, Tool
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.workflow import Workflow, WorkflowStatus, ChannelType
from ..core.state_manager import StateManager
from ..core.approval_engine import ApprovalEngine, ApprovalUISchemaGenerator, ApprovalValidator
from ..integrations.channels import ChannelManager
from ..ui.approval_ui import ApprovalUIRenderer

logger = structlog.get_logger(__name__)


class ApprovalRequiredTool(Tool):
    """
    SmolAgent tool that requests human approval before proceeding.
    """
    
    name = "request_approval"
    description = """
    Request human approval before proceeding with an action. Use this when the agent
    needs human confirmation, feedback, or decision-making input before continuing.
    """
    inputs = {
        "title": {
            "type": "string", 
            "description": "Title of the approval request"
        },
        "description": {
            "type": "string",
            "description": "Detailed description of what needs approval"
        },
        "approval_type": {
            "type": "string",
            "description": "Type of approval: 'binary', 'form', 'budget'",
            "enum": ["binary", "form", "budget"]
        },
        "context_data": {
            "type": "object",
            "description": "Additional context data for the approval"
        },
        "timeout_hours": {
            "type": "number",
            "description": "Hours to wait for approval before timing out",
            "default": 24
        },
        "required_approver": {
            "type": "string",
            "description": "Email/ID of the required approver"
        }
    }
    output_type = "object"
    
    def __init__(self, orchestrator: 'SmolAgentOrchestrator', **kwargs):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
    
    def forward(
        self, 
        title: str,
        description: str,
        approval_type: str = "binary",
        context_data: Dict[str, Any] = None,
        timeout_hours: float = 24,
        required_approver: str = "admin"
    ) -> Dict[str, Any]:
        """Request approval from human."""
        return asyncio.run(self._async_forward(
            title, description, approval_type, 
            context_data or {}, timeout_hours, required_approver
        ))
    
    async def _async_forward(
        self,
        title: str,
        description: str,
        approval_type: str,
        context_data: Dict[str, Any],
        timeout_hours: float,
        required_approver: str
    ) -> Dict[str, Any]:
        """Async implementation of approval request."""
        try:
            # Request approval through orchestrator
            approval_response = await self.orchestrator.request_human_approval(
                title=title,
                description=description,
                approval_type=approval_type,
                context_data=context_data,
                timeout_hours=timeout_hours,
                requested_from=required_approver
            )
            
            return approval_response
            
        except Exception as e:
            logger.error("Error in approval tool", error=str(e))
            return {
                "approved": False,
                "error": str(e),
                "message": f"Approval request failed: {str(e)}"
            }


class CheckpointTool(Tool):
    """
    SmolAgent tool that creates state checkpoints for rollback.
    """
    
    name = "create_checkpoint"
    description = """
    Create a checkpoint of the current workflow state for potential rollback.
    Use this before critical operations or at major workflow milestones.
    """
    inputs = {
        "checkpoint_name": {
            "type": "string",
            "description": "Name for the checkpoint"
        },
        "description": {
            "type": "string", 
            "description": "Description of what state is being checkpointed"
        },
        "step_name": {
            "type": "string",
            "description": "Current workflow step name"
        }
    }
    output_type = "object"
    
    def __init__(self, orchestrator: 'SmolAgentOrchestrator', **kwargs):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
    
    def forward(
        self, 
        checkpoint_name: str,
        description: str,
        step_name: str
    ) -> Dict[str, Any]:
        """Create checkpoint."""
        return asyncio.run(self._async_forward(checkpoint_name, description, step_name))
    
    async def _async_forward(
        self,
        checkpoint_name: str,
        description: str,
        step_name: str
    ) -> Dict[str, Any]:
        """Async implementation of checkpoint creation."""
        try:
            checkpoint = await self.orchestrator.create_checkpoint(
                checkpoint_name, step_name, description
            )
            
            return {
                "success": True,
                "checkpoint_id": str(checkpoint.id),
                "checkpoint_name": checkpoint_name,
                "message": f"Checkpoint '{checkpoint_name}' created successfully"
            }
            
        except Exception as e:
            logger.error("Error creating checkpoint", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create checkpoint: {str(e)}"
            }


class RollbackTool(Tool):
    """
    SmolAgent tool that rolls back workflow to a previous checkpoint.
    """
    
    name = "rollback_to_checkpoint"
    description = """
    Roll back the workflow to a previous checkpoint. Use this when something
    goes wrong and you need to return to a known good state.
    """
    inputs = {
        "checkpoint_name": {
            "type": "string",
            "description": "Name of the checkpoint to roll back to (optional - uses latest if not provided)"
        },
        "reason": {
            "type": "string",
            "description": "Reason for the rollback"
        }
    }
    output_type = "object"
    
    def __init__(self, orchestrator: 'SmolAgentOrchestrator', **kwargs):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
    
    def forward(self, checkpoint_name: str = None, reason: str = "") -> Dict[str, Any]:
        """Rollback to checkpoint."""
        return asyncio.run(self._async_forward(checkpoint_name, reason))
    
    async def _async_forward(self, checkpoint_name: str, reason: str) -> Dict[str, Any]:
        """Async implementation of rollback."""
        try:
            workflow = await self.orchestrator.rollback_workflow(checkpoint_name, reason)
            
            return {
                "success": True,
                "checkpoint_name": workflow.rollback_point,
                "current_state": workflow.current_state,
                "message": f"Successfully rolled back to checkpoint: {workflow.rollback_point}"
            }
            
        except Exception as e:
            logger.error("Error during rollback", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Rollback failed: {str(e)}"
            }


class SmolAgentOrchestrator:
    """
    Main orchestrator that integrates SmolAgents with human-in-the-loop approvals,
    state management, and complete workflow orchestration capabilities.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        model_config: Optional[Dict[str, Any]] = None,
        agent_type: str = "code",  # "code" or "tool_calling"
        additional_tools: Optional[List[Tool]] = None
    ):
        # Core components
        self.db = db_session
        self.state_manager = StateManager(db_session)
        self.approval_engine = ApprovalEngine(db_session, self.state_manager)
        self.ui_renderer = ApprovalUIRenderer(self.approval_engine)
        self.channel_manager = ChannelManager(self.approval_engine, self.ui_renderer)
        
        # Current workflow context
        self.current_workflow: Optional[Workflow] = None
        self.current_step = 0
        
        # Logging
        self.logger = logger.bind(component="orchestrator")
        
        # Initialize SmolAgent
        self._initialize_agent(model_config, agent_type, additional_tools)
        
        # Register approval components
        self._register_approval_components()
        
        # Setup channel handlers for approval engine
        self._setup_channel_handlers()
    
    def _initialize_agent(
        self,
        model_config: Optional[Dict[str, Any]],
        agent_type: str,
        additional_tools: Optional[List[Tool]]
    ):
        """Initialize the SmolAgent."""
        # Setup model
        if model_config:
            if model_config.get("provider") == "inference_client":
                self.model = InferenceClientModel(
                    model_id=model_config.get("model_id", "Qwen/Qwen2.5-Coder-32B-Instruct")
                )
            # Add other model types as needed
        else:
            self.model = InferenceClientModel()  # Default model
        
        # Create core tools
        core_tools = [
            ApprovalRequiredTool(self),
            CheckpointTool(self),
            RollbackTool(self)
        ]
        
        # Add additional tools
        if additional_tools:
            core_tools.extend(additional_tools)
        
        # Initialize agent
        if agent_type == "code":
            self.agent = CodeAgent(
                tools=core_tools,
                model=self.model,
                max_steps=20,
                additional_authorized_imports=["time", "json", "uuid", "datetime"]
            )
        else:
            self.agent = ToolCallingAgent(
                tools=core_tools,
                model=self.model,
                max_steps=15
            )
    
    def _register_approval_components(self):
        """Register approval UI generators and validators."""
        # Register UI generators
        self.approval_engine.register_ui_generator(
            "binary", 
            ApprovalUISchemaGenerator.binary_approval
        )
        self.approval_engine.register_ui_generator(
            "form", 
            ApprovalUISchemaGenerator.form_approval
        )
        self.approval_engine.register_ui_generator(
            "budget", 
            ApprovalUISchemaGenerator.budget_approval
        )
        
        # Register validators
        self.approval_engine.register_validator(
            "binary", 
            ApprovalValidator.binary_validator
        )
        self.approval_engine.register_validator(
            "form", 
            ApprovalValidator.form_validator
        )
        self.approval_engine.register_validator(
            "budget", 
            ApprovalValidator.budget_validator
        )
    
    def _setup_channel_handlers(self):
        """Setup channel handlers for the approval engine."""
        # Register channel handlers with approval engine
        self.approval_engine.register_channel_handler(
            ChannelType.EMAIL,
            self.channel_manager.send_approval_notification
        )
        self.approval_engine.register_channel_handler(
            ChannelType.SLACK,
            self.channel_manager.send_approval_notification
        )
        self.approval_engine.register_channel_handler(
            ChannelType.WEBHOOK,
            self.channel_manager.send_approval_notification
        )
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        workflow_definition: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> Workflow:
        """Create a new workflow."""
        workflow = await self.state_manager.create_workflow(
            name=name,
            definition=workflow_definition,
            input_data=input_data,
            created_by=created_by,
            description=description
        )
        
        self.current_workflow = workflow
        self.logger.info("Workflow created", workflow_id=str(workflow.id), name=name)
        
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: Optional[UUID] = None,
        task: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow with SmolAgent."""
        try:
            # Get or use current workflow
            if workflow_id:
                self.current_workflow = await self.state_manager.get_workflow(workflow_id)
            
            if not self.current_workflow:
                raise ValueError("No workflow available for execution")
            
            # Transition to running state
            await self.state_manager.transition_workflow_status(
                self.current_workflow.id,
                WorkflowStatus.RUNNING,
                triggered_by="agent",
                reason="Workflow execution started"
            )
            
            # Prepare execution context
            execution_context = {
                "workflow_id": str(self.current_workflow.id),
                "workflow_name": self.current_workflow.name,
                "input_data": self.current_workflow.input_data,
                "current_state": self.current_workflow.current_state,
                **(context or {})
            }
            
            # Create initial checkpoint
            await self.create_checkpoint(
                "execution_start",
                "execution_start",
                "Workflow execution started"
            )
            
            # Execute with SmolAgent
            if task:
                # Execute specific task
                enhanced_task = self._enhance_task_with_context(task, execution_context)
                result = self.agent.run(enhanced_task)
            else:
                # Execute workflow definition
                result = await self._execute_workflow_steps()
            
            # Update workflow state
            await self._update_workflow_completion(result)
            
            return {
                "success": True,
                "workflow_id": str(self.current_workflow.id),
                "result": result,
                "final_state": self.current_workflow.current_state
            }
            
        except Exception as e:
            self.logger.error("Workflow execution failed", error=str(e))
            
            if self.current_workflow:
                await self.state_manager.transition_workflow_status(
                    self.current_workflow.id,
                    WorkflowStatus.FAILED,
                    triggered_by="agent",
                    reason=f"Execution failed: {str(e)}"
                )
            
            return {
                "success": False,
                "error": str(e),
                "workflow_id": str(self.current_workflow.id) if self.current_workflow else None
            }
    
    async def _execute_workflow_steps(self) -> Any:
        """Execute workflow steps defined in the workflow definition."""
        steps = self.current_workflow.definition.get("steps", [])
        results = []
        
        for i, step in enumerate(steps):
            self.current_step = i
            
            # Update workflow state
            step_state = {
                **self.current_workflow.current_state,
                "step_index": i,
                "step_name": step.get("name", f"step_{i}"),
                "step_data": step
            }
            
            await self.state_manager.update_workflow_state(
                self.current_workflow.id,
                step_state,
                step.get("name", f"step_{i}"),
                triggered_by="agent"
            )
            
            # Create checkpoint for critical steps
            if step.get("create_checkpoint", False):
                await self.create_checkpoint(
                    f"step_{i}_{step.get('name', 'unnamed')}",
                    step.get("name", f"step_{i}"),
                    f"Checkpoint before step: {step.get('description', 'No description')}"
                )
            
            # Execute step
            step_result = await self._execute_single_step(step)
            results.append({
                "step": i,
                "name": step.get("name"),
                "result": step_result
            })
            
            # Check if step requires approval
            if step.get("requires_approval", False):
                approval_result = await self._handle_step_approval(step, step_result)
                if not approval_result.get("approved", False):
                    raise Exception(f"Step {i} was not approved: {approval_result.get('message', 'Unknown reason')}")
        
        return results
    
    async def _execute_single_step(self, step: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        step_type = step.get("type", "task")
        
        if step_type == "task":
            # Execute task with SmolAgent
            task = step.get("task", "")
            enhanced_task = self._enhance_task_with_context(task, step.get("context", {}))
            return self.agent.run(enhanced_task)
        
        elif step_type == "approval":
            # Direct approval request
            return await self.request_human_approval(
                title=step.get("title", "Approval Required"),
                description=step.get("description", ""),
                approval_type=step.get("approval_type", "binary"),
                context_data=step.get("context_data", {}),
                requested_from=step.get("requested_from", "admin")
            )
        
        elif step_type == "checkpoint":
            # Create checkpoint
            checkpoint = await self.create_checkpoint(
                step.get("checkpoint_name", f"step_checkpoint_{self.current_step}"),
                step.get("name", f"step_{self.current_step}"),
                step.get("description", "Step checkpoint")
            )
            return {"checkpoint_created": True, "checkpoint_id": str(checkpoint.id)}
        
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    async def _handle_step_approval(self, step: Dict[str, Any], step_result: Any) -> Dict[str, Any]:
        """Handle approval requirements for a step."""
        approval_config = step.get("approval_config", {})
        
        return await self.request_human_approval(
            title=approval_config.get("title", f"Approve Step: {step.get('name', 'Unnamed')}"),
            description=approval_config.get("description", f"Please approve the result of step: {step_result}"),
            approval_type=approval_config.get("type", "binary"),
            context_data={
                "step_name": step.get("name"),
                "step_result": step_result,
                **approval_config.get("context_data", {})
            },
            requested_from=approval_config.get("requested_from", "admin")
        )
    
    def _enhance_task_with_context(self, task: str, context: Dict[str, Any]) -> str:
        """Enhance task with workflow context and available tools."""
        context_str = json.dumps(context, indent=2) if context else "{}"
        
        enhanced_task = f"""
{task}

WORKFLOW CONTEXT:
{context_str}

AVAILABLE HUMAN-IN-LOOP TOOLS:
- request_approval: Request human approval before proceeding with critical actions
- create_checkpoint: Create state checkpoints for rollback capability  
- rollback_to_checkpoint: Rollback to a previous checkpoint if needed

GUIDELINES:
- Use request_approval for any action that could have significant impact
- Create checkpoints before major operations
- Include context and clear descriptions in approval requests
- Handle approval rejections gracefully by explaining alternatives
        """.strip()
        
        return enhanced_task
    
    async def request_human_approval(
        self,
        title: str,
        description: str,
        approval_type: str = "binary",
        context_data: Optional[Dict[str, Any]] = None,
        timeout_hours: float = 24,
        requested_from: str = "admin",
        channels: Optional[List[ChannelType]] = None
    ) -> Dict[str, Any]:
        """Request human approval and wait for response."""
        if not self.current_workflow:
            raise ValueError("No active workflow for approval request")
        
        # Create approval request
        approval = await self.approval_engine.request_approval(
            workflow_id=self.current_workflow.id,
            title=title,
            description=description,
            approval_type=approval_type,
            requested_from=requested_from,
            requested_by="agent",
            channels=channels or [ChannelType.UI, ChannelType.EMAIL],
            context_data=context_data or {},
            timeout_hours=int(timeout_hours)
        )
        
        # Wait for approval response
        return await self._wait_for_approval(approval.id, timeout_hours)
    
    async def _wait_for_approval(
        self, 
        approval_id: UUID, 
        timeout_hours: float
    ) -> Dict[str, Any]:
        """Wait for approval response with timeout."""
        timeout_seconds = timeout_hours * 3600
        start_time = datetime.now(timezone.utc)
        
        while True:
            # Check if approval has been responded to
            approval = await self.approval_engine.get_approval(approval_id)
            if not approval:
                return {
                    "approved": False,
                    "error": "Approval request not found",
                    "decision": "error"
                }
            
            if approval.status.value != "pending":
                # Approval has been processed
                return {
                    "approved": approval.status.value == "approved",
                    "decision": approval.decision or approval.status.value,
                    "response_data": approval.response_data,
                    "feedback": approval.responses[0].feedback if approval.responses else None,
                    "approver": approval.responses[0].approver_id if approval.responses else None
                }
            
            # Check timeout
            elapsed = datetime.now(timezone.utc) - start_time
            if elapsed.total_seconds() > timeout_seconds:
                await self.approval_engine.expire_approval(approval_id)
                return {
                    "approved": False,
                    "error": "Approval request timed out",
                    "decision": "expired"
                }
            
            # Wait before checking again
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def create_checkpoint(
        self,
        checkpoint_name: str,
        step_name: str,
        description: str = ""
    ) -> Any:
        """Create a workflow checkpoint."""
        if not self.current_workflow:
            raise ValueError("No active workflow for checkpoint")
        
        return await self.state_manager.create_checkpoint(
            self.current_workflow.id,
            checkpoint_name,
            step_name,
            description,
            created_by="agent"
        )
    
    async def rollback_workflow(
        self,
        checkpoint_name: Optional[str] = None,
        reason: str = ""
    ) -> Workflow:
        """Rollback workflow to a checkpoint."""
        if not self.current_workflow:
            raise ValueError("No active workflow for rollback")
        
        workflow = await self.state_manager.rollback_workflow(
            self.current_workflow.id,
            checkpoint_name,
            triggered_by="agent",
            reason=reason
        )
        
        self.current_workflow = workflow
        return workflow
    
    async def _update_workflow_completion(self, result: Any):
        """Update workflow upon completion."""
        if not self.current_workflow:
            return
        
        # Update final state
        final_state = {
            **self.current_workflow.current_state,
            "completed": True,
            "final_result": result,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        await self.state_manager.update_workflow_state(
            self.current_workflow.id,
            final_state,
            "completed",
            triggered_by="agent"
        )
        
        # Transition to completed status
        await self.state_manager.transition_workflow_status(
            self.current_workflow.id,
            WorkflowStatus.COMPLETED,
            triggered_by="agent",
            reason="Workflow execution completed successfully"
        )
    
    def configure_slack_integration(
        self,
        bot_token: str,
        signing_secret: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Configure Slack integration."""
        self.channel_manager.register_slack_handler(
            bot_token=bot_token,
            signing_secret=signing_secret,
            config=config
        )
        self.logger.info("Slack integration configured")
    
    def configure_email_integration(
        self,
        sendgrid_api_key: Optional[str] = None,
        smtp_config: Optional[Dict[str, Any]] = None
    ):
        """Configure email integration."""
        self.channel_manager.register_email_handler(
            sendgrid_api_key=sendgrid_api_key,
            smtp_config=smtp_config
        )
        self.logger.info("Email integration configured")
    
    def configure_webhook_integration(self):
        """Configure webhook integration."""
        self.channel_manager.register_webhook_handler()
        self.logger.info("Webhook integration configured")
    
    def get_slack_app(self):
        """Get Slack app for web server integration."""
        return self.channel_manager.get_slack_app()
    
    async def get_workflow_status(self, workflow_id: UUID) -> Dict[str, Any]:
        """Get comprehensive workflow status."""
        workflow = await self.state_manager.get_workflow(
            workflow_id, 
            include_events=True, 
            include_approvals=True,
            include_checkpoints=True
        )
        
        if not workflow:
            return {"error": "Workflow not found"}
        
        return {
            "workflow_id": str(workflow.id),
            "name": workflow.name,
            "status": workflow.status.value,
            "current_step": workflow.current_step,
            "progress": f"{workflow.current_state.get('step_index', 0)}/{workflow.total_steps}" if workflow.total_steps else "Unknown",
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
            "current_state": workflow.current_state,
            "pending_approvals": [
                {
                    "id": str(approval.id),
                    "title": approval.title,
                    "status": approval.status.value,
                    "expires_at": approval.expires_at.isoformat()
                }
                for approval in workflow.approvals 
                if approval.status.value == "pending"
            ],
            "recent_events": [
                {
                    "type": event.event_type.value,
                    "description": event.description,
                    "created_at": event.created_at.isoformat()
                }
                for event in workflow.events[-5:]  # Last 5 events
            ] if workflow.events else [],
            "checkpoints": [
                {
                    "name": checkpoint.checkpoint_name,
                    "step": checkpoint.step_name,
                    "created_at": checkpoint.created_at.isoformat()
                }
                for checkpoint in workflow.checkpoints[-3:]  # Last 3 checkpoints
            ] if workflow.checkpoints else []
        }