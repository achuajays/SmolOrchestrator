# Human-in-the-Loop System with Complete State Management and Rollback

A comprehensive **event-driven, stateful orchestration system** that enables **SmolAgents** to pause execution, request human approvals through multiple channels, and resume automatically with complete state management and rollback capabilities.

## 🧩 Problem Statement & Solution

Modern agent systems often require **human approvals or feedback** before executing critical actions — a purchase, a deployment, a message, a contract. This system solves the challenge of creating **asynchronous**, **multi-channel**, and **stateful** approval loops that remain **event-driven**, **state-aware**, and **resilient to failures** even when approvals happen hours later or through different channels.

## ✨ Key Features

### 🤖 Agent Integration
- **SmolAgent Integration**: Seamless integration with Hugging Face SmolAgents framework
- **Custom Tools**: Built-in approval, checkpoint, and rollback tools for agents
- **Enhanced Task Context**: Automatic context enrichment with workflow state and available tools
- **Multi-Agent Support**: Support for both CodeAgent and ToolCallingAgent types

### 🔄 State Management & Rollback
- **Complete State Tracking**: PostgreSQL-backed state management with full audit trail
- **Automatic Checkpoints**: Configurable checkpoint creation at workflow milestones
- **Rollback Capability**: Rollback to any checkpoint with complete state restoration
- **Event-Driven Architecture**: Comprehensive event logging for all state transitions

### 👥 Human-in-the-Loop Approvals
- **Multi-Channel Notifications**: Email, Slack, webhooks, and UI notifications
- **Dynamic UI Generation**: Configurable approval forms from JSON schemas
- **Multiple Approval Types**: Binary, form-based, and budget approvals
- **Multi-Approver Support**: Threshold-based approvals with multiple approvers
- **Timeout & Expiration**: Configurable timeouts with automatic expiration handling

### 🌐 Multi-Channel Integration
- **Slack Integration**: Rich interactive messages with approval buttons and commands
- **Email Support**: HTML emails with quick action buttons (SendGrid/SMTP)
- **Webhook Support**: External system integration via webhooks
- **Web UI**: Streamlit and Gradio-based approval interfaces

### 🛡️ Resilience & Recovery
- **Retry Logic**: Configurable retry mechanisms for failed operations  
- **Failure Recovery**: Automatic recovery for transient failures
- **Timeout Management**: Graceful handling of human response delays
- **Error Handling**: Comprehensive error handling with fallback strategies

### 📊 Observability
- **Structured Logging**: JSON-structured logs with contextual information
- **Audit Trail**: Complete workflow execution history
- **Real-time Status**: Live workflow status and progress tracking
- **Event Timeline**: Detailed event timeline for debugging and analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SmolAgent Orchestrator                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ ApprovalTool    │  │ CheckpointTool  │  │ RollbackTool    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
└─────────────────────┬───────────────────────────────────┬─────────┘
                      │                                   │
┌─────────────────────▼─────────────────────┐ ┌──────────▼──────────┐
│            Approval Engine                │ │   State Manager     │
│  ┌─────────────────────────────────────┐  │ │  ┌───────────────┐  │
│  │         Channel Manager             │  │ │  │  PostgreSQL   │  │
│  │  ┌─────────┐ ┌─────────┐ ┌────────┐ │  │ │  │    Database   │  │
│  │  │  Slack  │ │  Email  │ │Webhook │ │  │ │  └───────────────┘  │
│  │  └─────────┘ └─────────┘ └────────┘ │  │ │  ┌───────────────┐  │
│  └─────────────────────────────────────┘  │ │  │   Event Log   │  │
│  ┌─────────────────────────────────────┐  │ │  └───────────────┘  │
│  │         UI Renderer                 │  │ │  ┌───────────────┐  │
│  │  ┌─────────┐ ┌─────────┐ ┌────────┐ │  │ │  │  Checkpoints  │  │
│  │  │Streamlit│ │ Gradio  │ │  HTML  │ │  │ │  └───────────────┘  │
│  │  └─────────┘ └─────────┘ └────────┘ │  │ └─────────────────────┘
│  └─────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/achuajays/SmolOrchestrator.git
cd SmolOrchestrator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Basic Usage

```python
import asyncio
from src.agents.orchestrator import SmolAgentOrchestrator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

async def main():
    # Database setup
    engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)
    
    async with AsyncSessionLocal() as session:
        # Create orchestrator
        orchestrator = SmolAgentOrchestrator(
            db_session=session,
            model_config={"provider": "inference_client"},
            agent_type="code"
        )
        
        # Configure integrations
        orchestrator.configure_slack_integration(
            bot_token="xoxb-your-token",
            signing_secret="your-secret"
        )
        
        orchestrator.configure_email_integration(
            sendgrid_api_key="your-api-key"
        )
        
        # Create and execute workflow
        workflow = await orchestrator.create_workflow(
            name="Demo Workflow",
            description="Example workflow with approvals",
            workflow_definition={
                "steps": [
                    {
                        "name": "critical_task",
                        "type": "task", 
                        "task": "Perform a critical operation that needs approval",
                        "requires_approval": True,
                        "approval_config": {
                            "title": "Critical Operation Approval",
                            "description": "Please approve this critical operation",
                            "type": "binary",
                            "requested_from": "admin@company.com"
                        }
                    }
                ]
            }
        )
        
        result = await orchestrator.execute_workflow()
        print("Workflow completed:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent Tools Usage

Your SmolAgents automatically get access to approval tools:

```python
# In your agent task
task = """
I need to delete important files. Let me request approval first.

Use request_approval tool with:
- title: "File Deletion Approval"  
- description: "Need to delete 100GB of old backup files"
- approval_type: "binary"
- required_approver: "admin@company.com"

After approval, create a checkpoint before proceeding.
"""

result = agent.run(task)
```

## 📋 Workflow Definition Format

Workflows are defined using a JSON structure:

```json
{
  "name": "Example Workflow",
  "description": "Workflow description",
  "steps": [
    {
      "name": "step_name",
      "type": "task|approval|checkpoint",
      "task": "Task description for agent",
      "create_checkpoint": true,
      "requires_approval": true,
      "approval_config": {
        "title": "Approval Title",
        "description": "What needs approval",
        "type": "binary|form|budget", 
        "requested_from": "approver@company.com",
        "context_data": {
          "additional": "context"
        }
      },
      "context": {
        "step_specific": "data"
      }
    }
  ]
}
```

## 🎯 Approval Types

### Binary Approval
Simple approve/reject decisions:

```json
{
  "approval_type": "binary",
  "title": "Simple Approval",
  "description": "Please approve or reject this action"
}
```

### Form-Based Approval
Custom form fields for complex decisions:

```json
{
  "approval_type": "form",
  "context_data": {
    "form_fields": [
      {
        "name": "priority",
        "type": "string",
        "title": "Priority Level",
        "enum": ["low", "medium", "high"],
        "required": true
      },
      {
        "name": "comments",
        "type": "string", 
        "title": "Additional Comments",
        "format": "textarea"
      }
    ]
  }
}
```

### Budget Approval
Financial approval with amount validation:

```json
{
  "approval_type": "budget",
  "context_data": {
    "amount": 50000,
    "currency": "USD",
    "department": "marketing"
  }
}
```

## 🔧 Integration Configuration

### Slack Integration

```python
orchestrator.configure_slack_integration(
    bot_token="xoxb-your-bot-token",
    signing_secret="your-signing-secret"
)

# The system automatically creates interactive Slack messages
# with approval buttons and /approvals command support
```

### Email Integration

```python
# Using SendGrid
orchestrator.configure_email_integration(
    sendgrid_api_key="your-sendgrid-key"
)

# Using SMTP
orchestrator.configure_email_integration(
    smtp_config={
        "host": "smtp.gmail.com",
        "port": 587,
        "use_tls": True,
        "username": "your-email@gmail.com",
        "password": "your-password",
        "from_email": "noreply@company.com"
    }
)
```

### Webhook Integration

```python
orchestrator.configure_webhook_integration()

# Configure webhook URLs in approval requests:
# {
#   "channels": ["webhook"],
#   "channel_configs": {
#     "webhook": {
#       "url": "https://your-system.com/approval-webhook",
#       "headers": {"Authorization": "Bearer token"}
#     }
#   }
# }
```

## 🗄️ Database Schema

The system uses PostgreSQL with the following key tables:

- **workflows**: Main workflow state and metadata
- **approval_requests**: Human approval requests
- **approval_responses**: Individual approval responses  
- **workflow_events**: Complete audit trail
- **workflow_checkpoints**: State snapshots for rollback
- **channel_configurations**: Communication channel configs

## 📊 Monitoring & Observability

### Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)
logger.info("Workflow started", workflow_id="123", user="admin")
```

### Workflow Status API

```python
status = await orchestrator.get_workflow_status(workflow_id)
# Returns comprehensive status including:
# - Current step and progress
# - Pending approvals
# - Recent events  
# - Available checkpoints
```

### Event Timeline

All workflow events are logged with:
- Event type and description
- Before/after state snapshots
- Triggered by information
- Contextual metadata

## 🛡️ Error Handling & Recovery

### Automatic Retry

```python
class WorkflowRecoveryService:
    async def recover_failed_workflows(self):
        # Automatically attempts to recover failed workflows
        # based on failure type and retry policies
```

### Manual Rollback

```python
# Rollback to specific checkpoint
workflow = await orchestrator.rollback_workflow(
    checkpoint_name="pre_deployment",
    reason="Deployment failed validation"
)

# Rollback to latest checkpoint
workflow = await orchestrator.rollback_workflow(
    reason="General failure recovery"
)
```

## 📝 Example Workflows

### E-commerce Order Processing

```python
workflow_definition = {
    "name": "Order Processing",
    "steps": [
        {
            "name": "validate_order",
            "type": "task",
            "task": "Validate order details and inventory",
            "create_checkpoint": True
        },
        {
            "name": "payment_approval", 
            "type": "task",
            "task": "Process payment",
            "requires_approval": True,
            "approval_config": {
                "title": "High-Value Payment",
                "type": "budget",
                "requested_from": "finance@company.com"
            }
        }
    ]
}
```

### Deployment Pipeline

```python
workflow_definition = {
    "name": "Production Deployment",
    "steps": [
        {
            "name": "pre_deployment_checks",
            "type": "task", 
            "task": "Run tests and security scans",
            "create_checkpoint": True
        },
        {
            "name": "deployment_approval",
            "type": "approval",
            "title": "Production Deploy Approval",
            "approval_type": "form",
            "requested_from": "devops@company.com"
        },
        {
            "name": "deploy",
            "type": "task",
            "task": "Deploy to production",
            "create_checkpoint": True
        }
    ]
}
```

## 🏃‍♂️ Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run the comprehensive demo
python examples/demo_application.py
```

The demo showcases:
- ✅ E-commerce order processing with approvals
- ✅ Multi-level budget approval workflow  
- ✅ Deployment pipeline with rollback
- ✅ Document review with multi-approvers
- ✅ Interactive approval handling

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📚 API Reference

### SmolAgentOrchestrator

Main orchestrator class integrating all components.

#### Methods

- `create_workflow(name, description, workflow_definition, input_data, created_by)`: Create new workflow
- `execute_workflow(workflow_id, task, context)`: Execute workflow  
- `request_human_approval(title, description, approval_type, ...)`: Request approval
- `create_checkpoint(checkpoint_name, step_name, description)`: Create checkpoint
- `rollback_workflow(checkpoint_name, reason)`: Rollback to checkpoint
- `get_workflow_status(workflow_id)`: Get comprehensive status

### ApprovalEngine

Core approval handling engine.

#### Methods
- `request_approval(workflow_id, title, description, ...)`: Create approval request
- `respond_to_approval(approval_id, approver_id, decision, ...)`: Process response
- `expire_approval(approval_id)`: Manually expire approval
- `list_pending_approvals(requested_from, workflow_id)`: List pending approvals

### StateManager  

Workflow state management with PostgreSQL backend.

#### Methods
- `create_workflow(name, definition, input_data, ...)`: Create workflow
- `update_workflow_state(workflow_id, new_state, step_name, ...)`: Update state
- `transition_workflow_status(workflow_id, new_status, ...)`: Change status
- `create_checkpoint(workflow_id, checkpoint_name, ...)`: Create checkpoint
- `rollback_workflow(workflow_id, checkpoint_name, ...)`: Rollback workflow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For support and questions:
- Create an issue on GitHub
- Check the documentation and examples
- Review the comprehensive demo application

---

**Built with ❤️ using Hugging Face SmolAgents, PostgreSQL, and modern async Python**
