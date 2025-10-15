# SmolAgent Orchestrator - Detailed Architecture

This chart shows the detailed architecture of the SmolAgent Orchestrator component (`src/agents/orchestrator.py`).

```mermaid
graph TB
    %% Input Sources
    User[👤 User Request<br/>- Workflow definition<br/>- Task description<br/>- Context data]
    
    %% Main Orchestrator Class
    subgraph Orchestrator[🎭 SmolAgentOrchestrator Class]
        
        %% Core Components
        Agent[🤖 SmolAgent Instance<br/>- CodeAgent or ToolCallingAgent<br/>- Model configuration<br/>- Tool integration<br/>- Max steps configuration]
        
        WorkflowManager[📋 Workflow Manager<br/>- Workflow creation<br/>- Step execution<br/>- Progress tracking<br/>- Result collection]
        
        ContextEnhancer[🔍 Context Enhancer<br/>- Task enhancement<br/>- Context injection<br/>- Tool availability<br/>- Guidelines addition]
        
        %% Tool Registry
        subgraph ToolRegistry[🔧 Built-in Tools Registry]
            ApprovalTool[🤝 ApprovalRequiredTool<br/>**Purpose**: Request human approval<br/>**Inputs**: title, description, type<br/>**Logic**: Creates approval request<br/>**Returns**: Approval response data]
            
            CheckpointTool[📍 CheckpointTool<br/>**Purpose**: Create state snapshots<br/>**Inputs**: name, description, step<br/>**Logic**: Calls state manager<br/>**Returns**: Checkpoint metadata]
            
            RollbackTool[⏪ RollbackTool<br/>**Purpose**: Restore previous state<br/>**Inputs**: checkpoint name, reason<br/>**Logic**: State restoration<br/>**Returns**: Rollback confirmation]
        end
        
        %% Execution Engine
        subgraph ExecutionEngine[⚙️ Execution Engine]
            WorkflowExec[📝 Workflow Execution<br/>**Method**: execute_workflow()<br/>**Logic**: Step-by-step processing<br/>**Features**: Error handling, state updates]
            
            StepExec[🔄 Step Execution<br/>**Method**: _execute_single_step()<br/>**Types**: task, approval, checkpoint<br/>**Logic**: Type-specific handling]
            
            ApprovalHandler[✋ Approval Handler<br/>**Method**: _handle_step_approval()<br/>**Logic**: Approval requirement check<br/>**Features**: Async waiting, timeout]
            
            ContextInjector[💉 Context Injector<br/>**Method**: _enhance_task_with_context()<br/>**Logic**: Context string building<br/>**Features**: Tool descriptions, guidelines]
        end
        
        %% State Integration
        subgraph StateIntegration[📊 State Integration]
            StateSync[🔄 State Synchronization<br/>**Method**: create_workflow()<br/>**Logic**: Initial state creation<br/>**Features**: Metadata tracking]
            
            ProgressTracker[📈 Progress Tracking<br/>**Method**: update_workflow_state()<br/>**Logic**: Step progress updates<br/>**Features**: Real-time monitoring]
            
            CompletionHandler[✅ Completion Handler<br/>**Method**: _update_workflow_completion()<br/>**Logic**: Final state updates<br/>**Features**: Result storage]
        end
    end
    
    %% External Dependencies
    StateManager[📊 State Manager<br/>- Workflow state persistence<br/>- Event logging<br/>- Checkpoint management]
    
    ApprovalEngine[✋ Approval Engine<br/>- Approval request creation<br/>- Response handling<br/>- Multi-channel notifications]
    
    ChannelManager[📡 Channel Manager<br/>- Slack integration<br/>- Email notifications<br/>- Webhook delivery]
    
    Database[(🗄️ Database<br/>- Workflow records<br/>- State snapshots<br/>- Event audit trail)]
    
    HFModel[🤗 HuggingFace Model<br/>- InferenceClientModel<br/>- Model configuration<br/>- API integration]
    
    %% Data Flow
    User -->|Workflow Request| Orchestrator
    
    %% Internal Flows
    WorkflowManager --> ExecutionEngine
    ExecutionEngine --> ContextEnhancer
    ContextEnhancer --> Agent
    Agent --> ToolRegistry
    
    WorkflowExec --> StepExec
    StepExec --> ApprovalHandler
    ApprovalHandler --> ContextInjector
    
    StateIntegration --> StateSync
    StateSync --> ProgressTracker
    ProgressTracker --> CompletionHandler
    
    %% Tool Flows
    ApprovalTool -->|Approval Request| ApprovalEngine
    CheckpointTool -->|Create Checkpoint| StateManager
    RollbackTool -->|Rollback State| StateManager
    
    %% External Integration
    Orchestrator --> StateManager
    Orchestrator --> ApprovalEngine
    ApprovalEngine --> ChannelManager
    StateManager --> Database
    Agent --> HFModel
    
    %% Configuration Flow
    subgraph Config[⚙️ Configuration System]
        ModelConfig[🔧 Model Configuration<br/>**Structure**: Dictionary<br/>**Keys**: provider, model_id<br/>**Purpose**: LLM setup]
        
        AgentConfig[🤖 Agent Configuration<br/>**Types**: CodeAgent, ToolCallingAgent<br/>**Parameters**: max_steps, imports<br/>**Purpose**: Agent behavior]
        
        ToolConfig[🔧 Tool Configuration<br/>**Registry**: Built-in tools<br/>**Extension**: Additional tools<br/>**Purpose**: Capability expansion]
        
        ChannelConfig[📡 Channel Configuration<br/>**Slack**: Bot token, secrets<br/>**Email**: SMTP/SendGrid config<br/>**Purpose**: Multi-channel setup]
    end
    
    Config --> Orchestrator
    
    %% Method Details
    subgraph Methods[📋 Key Methods Explained]
        CreateWF[create_workflow()<br/>**Purpose**: Initialize new workflow<br/>**Steps**: 1. Create DB record<br/>2. Set initial state<br/>3. Log creation event<br/>**Returns**: Workflow object]
        
        ExecWF[execute_workflow()<br/>**Purpose**: Run workflow steps<br/>**Steps**: 1. Set running status<br/>2. Process each step<br/>3. Handle approvals<br/>4. Update completion<br/>**Returns**: Execution result]
        
        RequestApproval[request_human_approval()<br/>**Purpose**: Pause for human input<br/>**Steps**: 1. Create approval request<br/>2. Send notifications<br/>3. Wait for response<br/>4. Resume execution<br/>**Returns**: Approval data]
        
        CreateCheckpoint[create_checkpoint()<br/>**Purpose**: Save state snapshot<br/>**Steps**: 1. Capture current state<br/>2. Store in database<br/>3. Log checkpoint event<br/>**Returns**: Checkpoint object]
        
        RollbackWF[rollback_workflow()<br/>**Purpose**: Restore previous state<br/>**Steps**: 1. Find target checkpoint<br/>2. Restore state data<br/>3. Cancel pending approvals<br/>4. Log rollback event<br/>**Returns**: Restored workflow]
    end
    
    %% Styling
    classDef orchestrator fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef tools fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef execution fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef state fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef config fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef methods fill:#fafafa,stroke:#424242,stroke-width:2px
    
    class Orchestrator,WorkflowManager,ContextEnhancer,Agent orchestrator
    class ToolRegistry,ApprovalTool,CheckpointTool,RollbackTool tools
    class ExecutionEngine,WorkflowExec,StepExec,ApprovalHandler,ContextInjector execution
    class StateIntegration,StateSync,ProgressTracker,CompletionHandler state
    class StateManager,ApprovalEngine,ChannelManager,Database,HFModel external
    class Config,ModelConfig,AgentConfig,ToolConfig,ChannelConfig config
    class Methods,CreateWF,ExecWF,RequestApproval,CreateCheckpoint,RollbackWF methods
```

## Key Components Explanation

### 🎭 **SmolAgentOrchestrator Class**
**Location**: `src/agents/orchestrator.py` (lines 234-794)
**Purpose**: Main orchestration class that integrates SmolAgents with human-in-the-loop capabilities

**Key Responsibilities**:
- Initialize SmolAgent (CodeAgent/ToolCallingAgent)
- Manage workflow lifecycle 
- Provide approval, checkpoint, and rollback tools
- Handle async approval waiting
- Integrate with state management system

### 🤖 **SmolAgent Integration**
**Method**: `_initialize_agent()` (lines 270-311)
**Purpose**: Set up SmolAgent with custom tools and model configuration

**Process**:
1. Configure model (InferenceClientModel by default)
2. Create core tools (approval, checkpoint, rollback)
3. Initialize CodeAgent or ToolCallingAgent
4. Set max steps and authorized imports

### 🔧 **Built-in Tools**
**ApprovalRequiredTool** (lines 25-113):
- Requests human approval before proceeding
- Supports different approval types (binary, form, budget)
- Handles timeout and multi-approver scenarios
- Returns approval response data to agent

**CheckpointTool** (lines 116-180):
- Creates state snapshots for rollback
- Captures complete workflow state
- Stores execution context
- Enables recovery from failures

**RollbackTool** (lines 183-231):
- Restores workflow to previous checkpoint
- Handles state restoration
- Cancels pending operations
- Logs rollback reasons

### ⚙️ **Execution Engine**
**Workflow Execution** (lines 381-454):
- Processes workflow steps sequentially
- Handles different step types (task, approval, checkpoint)
- Manages error conditions and retries
- Updates state after each step

**Context Enhancement** (lines 551-573):
- Enriches agent tasks with workflow context
- Adds tool descriptions and guidelines  
- Provides execution state information
- Enables context-aware agent decisions

### 📊 **State Integration**
**State Management** (lines 647-682):
- Creates and updates workflow records
- Manages state transitions
- Handles checkpoint creation
- Provides rollback capabilities

**Progress Tracking**:
- Real-time workflow status updates
- Step completion monitoring
- Event logging for audit trail
- Performance metrics collection