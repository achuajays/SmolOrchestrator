"""
Resilience and retry logic system for handling failures, timeouts,
and recovery scenarios in human approval workflows.
"""

import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import UUID
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.workflow import Workflow, WorkflowStatus, ApprovalStatus, EventType
from .state_manager import StateManager, WorkflowRecoveryService

logger = structlog.get_logger(__name__)


class RetryPolicy(str, Enum):
    """Retry policy types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


class FailureType(str, Enum):
    """Types of failures that can occur."""
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        policy: RetryPolicy = RetryPolicy.EXPONENTIAL,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_errors: List[FailureType] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.policy = policy
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_errors = retryable_errors or [
            FailureType.TIMEOUT,
            FailureType.NETWORK_ERROR,
            FailureType.SERVICE_UNAVAILABLE,
            FailureType.RATE_LIMIT
        ]
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.policy == RetryPolicy.LINEAR:
            delay = self.base_delay * attempt
        elif self.policy == RetryPolicy.EXPONENTIAL:
            delay = self.base_delay * (self.backoff_factor ** attempt)
        elif self.policy == RetryPolicy.FIBONACCI:
            delay = self.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.base_delay
        
        # Apply maximum delay cap
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 2:
            return 1
        a, b = 1, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b
    
    def is_retryable(self, failure_type: FailureType) -> bool:
        """Check if error type is retryable."""
        return failure_type in self.retryable_errors


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for service protection.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = logger.bind(component="circuit_breaker")
    
    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker half-open, attempting reset")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() > self.timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(
                "Circuit breaker OPEN due to failures",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


class ResilientExecutor:
    """
    Resilient execution service with retry, timeout, and circuit breaker capabilities.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        default_retry_config: Optional[RetryConfig] = None,
        default_timeout: float = 300.0
    ):
        self.state_manager = state_manager
        self.default_retry_config = default_retry_config or RetryConfig()
        self.default_timeout = default_timeout
        self.circuit_breakers = {}
        self.logger = logger.bind(component="resilient_executor")
    
    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[Any]],
        *args,
        retry_config: Optional[RetryConfig] = None,
        timeout: Optional[float] = None,
        workflow_id: Optional[UUID] = None,
        operation_name: str = "operation",
        **kwargs
    ) -> Any:
        """Execute function with retry logic and resilience patterns."""
        config = retry_config or self.default_retry_config
        timeout_seconds = timeout or self.default_timeout
        
        for attempt in range(config.max_attempts):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                
                # Log successful execution
                if attempt > 0:
                    self.logger.info(
                        "Operation succeeded after retry",
                        operation=operation_name,
                        attempt=attempt + 1,
                        workflow_id=str(workflow_id) if workflow_id else None
                    )
                
                return result
            
            except asyncio.TimeoutError:
                failure_type = FailureType.TIMEOUT
                error_msg = f"Operation timed out after {timeout_seconds}s"
            
            except ConnectionError:
                failure_type = FailureType.NETWORK_ERROR
                error_msg = "Network connection error"
            
            except Exception as e:
                failure_type = self._classify_error(e)
                error_msg = str(e)
            
            # Check if error is retryable
            if not config.is_retryable(failure_type):
                self.logger.error(
                    "Non-retryable error encountered",
                    operation=operation_name,
                    error=error_msg,
                    failure_type=failure_type.value,
                    workflow_id=str(workflow_id) if workflow_id else None
                )
                raise
            
            # Check if this is the last attempt
            if attempt == config.max_attempts - 1:
                self.logger.error(
                    "All retry attempts exhausted",
                    operation=operation_name,
                    attempts=config.max_attempts,
                    error=error_msg,
                    workflow_id=str(workflow_id) if workflow_id else None
                )
                
                # Log retry failure event if workflow context
                if workflow_id:
                    await self.state_manager._create_event(
                        workflow_id,
                        EventType.RETRY_ATTEMPTED,
                        f"Retry attempts exhausted for {operation_name}",
                        event_data={
                            "operation": operation_name,
                            "max_attempts": config.max_attempts,
                            "final_error": error_msg,
                            "failure_type": failure_type.value
                        },
                        triggered_by="resilience_system"
                    )
                
                raise
            
            # Calculate delay and wait
            delay = config.calculate_delay(attempt)
            
            self.logger.warning(
                "Operation failed, retrying",
                operation=operation_name,
                attempt=attempt + 1,
                max_attempts=config.max_attempts,
                error=error_msg,
                failure_type=failure_type.value,
                retry_delay=delay,
                workflow_id=str(workflow_id) if workflow_id else None
            )
            
            # Log retry attempt event if workflow context
            if workflow_id:
                await self.state_manager._create_event(
                    workflow_id,
                    EventType.RETRY_ATTEMPTED,
                    f"Retrying {operation_name} (attempt {attempt + 1})",
                    event_data={
                        "operation": operation_name,
                        "attempt": attempt + 1,
                        "error": error_msg,
                        "failure_type": failure_type.value,
                        "retry_delay": delay
                    },
                    triggered_by="resilience_system"
                )
            
            await asyncio.sleep(delay)
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    async def execute_with_circuit_breaker(
        self,
        func: Callable[[], Awaitable[Any]],
        service_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(service_name)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for retry decision."""
        error_msg = str(error).lower()
        
        if "timeout" in error_msg:
            return FailureType.TIMEOUT
        elif "connection" in error_msg or "network" in error_msg:
            return FailureType.NETWORK_ERROR
        elif "unavailable" in error_msg or "503" in error_msg:
            return FailureType.SERVICE_UNAVAILABLE
        elif "rate limit" in error_msg or "429" in error_msg:
            return FailureType.RATE_LIMIT
        elif "validation" in error_msg or "400" in error_msg:
            return FailureType.VALIDATION_ERROR
        elif "authentication" in error_msg or "401" in error_msg:
            return FailureType.AUTHENTICATION_ERROR
        elif "permission" in error_msg or "403" in error_msg:
            return FailureType.PERMISSION_ERROR
        else:
            return FailureType.UNKNOWN_ERROR


class ApprovalTimeoutManager:
    """
    Manages approval timeouts and escalation logic.
    """
    
    def __init__(self, state_manager: StateManager, resilient_executor: ResilientExecutor):
        self.state_manager = state_manager
        self.resilient_executor = resilient_executor
        self.logger = logger.bind(component="timeout_manager")
        self._running_tasks = {}
    
    async def schedule_approval_timeout(
        self,
        approval_id: UUID,
        timeout_seconds: float,
        escalation_config: Optional[Dict[str, Any]] = None
    ):
        """Schedule timeout handling for approval request."""
        task = asyncio.create_task(
            self._handle_approval_timeout(approval_id, timeout_seconds, escalation_config)
        )
        self._running_tasks[approval_id] = task
        
        self.logger.info(
            "Scheduled approval timeout",
            approval_id=str(approval_id),
            timeout_seconds=timeout_seconds
        )
    
    def cancel_approval_timeout(self, approval_id: UUID):
        """Cancel scheduled timeout for approval."""
        if approval_id in self._running_tasks:
            self._running_tasks[approval_id].cancel()
            del self._running_tasks[approval_id]
            self.logger.info("Cancelled approval timeout", approval_id=str(approval_id))
    
    async def _handle_approval_timeout(
        self,
        approval_id: UUID,
        timeout_seconds: float,
        escalation_config: Optional[Dict[str, Any]]
    ):
        """Handle approval timeout and optional escalation."""
        try:
            # Wait for timeout period
            await asyncio.sleep(timeout_seconds)
            
            # Check if approval is still pending
            from ..core.approval_engine import ApprovalEngine  # Avoid circular import
            
            # This would be injected in real implementation
            approval_engine = None  # Would be passed in constructor
            
            if approval_engine:
                approval = await approval_engine.get_approval(approval_id)
                
                if approval and approval.status == ApprovalStatus.PENDING:
                    # Handle escalation if configured
                    if escalation_config:
                        await self._handle_escalation(approval_id, escalation_config)
                    else:
                        # Simple timeout expiration
                        await approval_engine.expire_approval(approval_id)
            
        except asyncio.CancelledError:
            self.logger.info("Approval timeout cancelled", approval_id=str(approval_id))
        except Exception as e:
            self.logger.error(
                "Error handling approval timeout",
                approval_id=str(approval_id),
                error=str(e)
            )
    
    async def _handle_escalation(
        self,
        approval_id: UUID,
        escalation_config: Dict[str, Any]
    ):
        """Handle approval escalation."""
        escalation_type = escalation_config.get("type", "notify")
        
        if escalation_type == "notify":
            # Send escalation notification
            escalation_recipients = escalation_config.get("recipients", [])
            # Implementation would send notifications to escalation recipients
            
        elif escalation_type == "auto_approve":
            # Automatically approve with escalation
            # Implementation would auto-approve the request
            pass
        
        elif escalation_type == "reassign":
            # Reassign to different approver
            new_approver = escalation_config.get("new_approver")
            # Implementation would reassign the approval
            
        self.logger.info(
            "Handled approval escalation",
            approval_id=str(approval_id),
            escalation_type=escalation_type
        )


class HealthChecker:
    """
    System health monitoring and automatic recovery.
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.logger = logger.bind(component="health_checker")
        self._health_checks = {}
        self._monitoring_task = None
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], Awaitable[bool]],
        interval_seconds: float = 30.0,
        timeout_seconds: float = 10.0
    ):
        """Register a health check."""
        self._health_checks[name] = {
            "func": check_func,
            "interval": interval_seconds,
            "timeout": timeout_seconds,
            "last_check": None,
            "status": "unknown",
            "consecutive_failures": 0
        }
        
        self.logger.info("Registered health check", name=name, interval=interval_seconds)
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_health())
            self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
            self.logger.info("Stopped health monitoring")
    
    async def _monitor_health(self):
        """Main health monitoring loop."""
        while True:
            try:
                for name, check_config in self._health_checks.items():
                    asyncio.create_task(self._run_health_check(name, check_config))
                
                # Wait before next monitoring cycle
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in health monitoring", error=str(e))
                await asyncio.sleep(5.0)
    
    async def _run_health_check(self, name: str, check_config: Dict[str, Any]):
        """Run individual health check."""
        now = datetime.now(timezone.utc)
        
        # Check if it's time to run this check
        if (check_config["last_check"] and 
            (now - check_config["last_check"]).total_seconds() < check_config["interval"]):
            return
        
        try:
            # Run health check with timeout
            is_healthy = await asyncio.wait_for(
                check_config["func"](),
                timeout=check_config["timeout"]
            )
            
            if is_healthy:
                if check_config["status"] != "healthy":
                    self.logger.info("Health check recovered", name=name)
                
                check_config["status"] = "healthy"
                check_config["consecutive_failures"] = 0
            else:
                self._handle_health_check_failure(name, check_config, "Check returned false")
        
        except asyncio.TimeoutError:
            self._handle_health_check_failure(name, check_config, "Timeout")
        except Exception as e:
            self._handle_health_check_failure(name, check_config, str(e))
        
        check_config["last_check"] = now
    
    def _handle_health_check_failure(self, name: str, check_config: Dict[str, Any], error: str):
        """Handle health check failure."""
        check_config["consecutive_failures"] += 1
        check_config["status"] = "unhealthy"
        
        self.logger.warning(
            "Health check failed",
            name=name,
            error=error,
            consecutive_failures=check_config["consecutive_failures"]
        )
        
        # Trigger alerts or recovery actions based on failure count
        if check_config["consecutive_failures"] >= 3:
            self.logger.error(
                "Health check critically unhealthy",
                name=name,
                consecutive_failures=check_config["consecutive_failures"]
            )
            # Could trigger automatic recovery actions here
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            name: {
                "status": config["status"],
                "last_check": config["last_check"].isoformat() if config["last_check"] else None,
                "consecutive_failures": config["consecutive_failures"]
            }
            for name, config in self._health_checks.items()
        }