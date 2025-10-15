"""
Comprehensive observability system providing monitoring, metrics,
tracing, and alerting for the human-in-the-loop system.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Deque
from uuid import UUID
from enum import Enum

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Info, start_http_server,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from opentelemetry import trace, baggage
from opentelemetry.trace import Tracer, Span, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.workflow import Workflow, WorkflowStatus, ApprovalStatus, EventType
from ..core.state_manager import StateManager

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Metric types for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    INFO = "info"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricsCollector:
    """
    Prometheus metrics collector for workflow and approval metrics.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = logger.bind(component="metrics_collector")
        
        # Workflow metrics
        self.workflows_created = Counter(
            'workflows_created_total',
            'Total number of workflows created',
            ['created_by', 'workflow_type'],
            registry=self.registry
        )
        
        self.workflows_completed = Counter(
            'workflows_completed_total',
            'Total number of workflows completed',
            ['status', 'workflow_type'],
            registry=self.registry
        )
        
        self.workflow_duration = Histogram(
            'workflow_duration_seconds',
            'Workflow execution duration in seconds',
            ['workflow_type', 'status'],
            registry=self.registry
        )
        
        self.active_workflows = Gauge(
            'active_workflows',
            'Number of currently active workflows',
            ['status'],
            registry=self.registry
        )
        
        # Approval metrics
        self.approvals_requested = Counter(
            'approvals_requested_total',
            'Total number of approval requests',
            ['approval_type', 'channel'],
            registry=self.registry
        )
        
        self.approvals_responded = Counter(
            'approvals_responded_total',
            'Total number of approval responses',
            ['decision', 'approval_type', 'channel'],
            registry=self.registry
        )
        
        self.approval_response_time = Histogram(
            'approval_response_time_seconds',
            'Time taken to respond to approvals',
            ['approval_type', 'decision'],
            registry=self.registry
        )
        
        self.pending_approvals = Gauge(
            'pending_approvals',
            'Number of pending approval requests',
            ['approval_type'],
            registry=self.registry
        )
        
        self.expired_approvals = Counter(
            'approvals_expired_total',
            'Total number of expired approvals',
            ['approval_type'],
            registry=self.registry
        )
        
        # System metrics
        self.system_errors = Counter(
            'system_errors_total',
            'Total number of system errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        self.retry_attempts = Counter(
            'retry_attempts_total',
            'Total number of retry attempts',
            ['operation', 'failure_type'],
            registry=self.registry
        )
        
        self.rollbacks_performed = Counter(
            'rollbacks_performed_total',
            'Total number of workflow rollbacks',
            ['reason'],
            registry=self.registry
        )
        
        # Channel-specific metrics
        self.channel_notifications = Counter(
            'channel_notifications_total',
            'Total notifications sent per channel',
            ['channel', 'success'],
            registry=self.registry
        )
        
        self.channel_response_time = Histogram(
            'channel_response_time_seconds',
            'Channel notification response time',
            ['channel'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'human_in_loop_system_info',
            'System information',
            registry=self.registry
        )
        
        # Set system info
        self.system_info.info({
            'version': '1.0.0',
            'component': 'human_in_loop_system'
        })
    
    def record_workflow_created(self, created_by: str, workflow_type: str = "default"):
        """Record workflow creation."""
        self.workflows_created.labels(
            created_by=created_by,
            workflow_type=workflow_type
        ).inc()
    
    def record_workflow_completed(self, status: str, workflow_type: str, duration_seconds: float):
        """Record workflow completion."""
        self.workflows_completed.labels(
            status=status,
            workflow_type=workflow_type
        ).inc()
        
        self.workflow_duration.labels(
            workflow_type=workflow_type,
            status=status
        ).observe(duration_seconds)
    
    def update_active_workflows(self, status_counts: Dict[str, int]):
        """Update active workflow gauges."""
        for status, count in status_counts.items():
            self.active_workflows.labels(status=status).set(count)
    
    def record_approval_request(self, approval_type: str, channels: List[str]):
        """Record approval request."""
        for channel in channels:
            self.approvals_requested.labels(
                approval_type=approval_type,
                channel=channel
            ).inc()
    
    def record_approval_response(
        self, 
        decision: str, 
        approval_type: str, 
        channel: str,
        response_time_seconds: float
    ):
        """Record approval response."""
        self.approvals_responded.labels(
            decision=decision,
            approval_type=approval_type,
            channel=channel
        ).inc()
        
        self.approval_response_time.labels(
            approval_type=approval_type,
            decision=decision
        ).observe(response_time_seconds)
    
    def update_pending_approvals(self, approval_type_counts: Dict[str, int]):
        """Update pending approval gauges."""
        for approval_type, count in approval_type_counts.items():
            self.pending_approvals.labels(approval_type=approval_type).set(count)
    
    def record_approval_expired(self, approval_type: str):
        """Record approval expiration."""
        self.expired_approvals.labels(approval_type=approval_type).inc()
    
    def record_system_error(self, error_type: str, component: str):
        """Record system error."""
        self.system_errors.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def record_retry_attempt(self, operation: str, failure_type: str):
        """Record retry attempt."""
        self.retry_attempts.labels(
            operation=operation,
            failure_type=failure_type
        ).inc()
    
    def record_rollback(self, reason: str):
        """Record workflow rollback."""
        self.rollbacks_performed.labels(reason=reason).inc()
    
    def record_channel_notification(self, channel: str, success: bool, response_time: float):
        """Record channel notification."""
        self.channel_notifications.labels(
            channel=channel,
            success=str(success).lower()
        ).inc()
        
        self.channel_response_time.labels(channel=channel).observe(response_time)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)


class TracingService:
    """
    OpenTelemetry-based distributed tracing service.
    """
    
    def __init__(
        self,
        service_name: str = "human_in_loop_system",
        jaeger_endpoint: Optional[str] = None
    ):
        self.service_name = service_name
        self.logger = logger.bind(component="tracing_service")
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        self.tracer: Tracer = trace.get_tracer(service_name)
        
        # Set up Jaeger exporter if endpoint provided
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.logger.info("Configured Jaeger tracing", endpoint=jaeger_endpoint)
    
    def start_span(
        self,
        name: str,
        workflow_id: Optional[UUID] = None,
        parent_span: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new tracing span."""
        span_attributes = attributes or {}
        
        if workflow_id:
            span_attributes["workflow.id"] = str(workflow_id)
        
        span = self.tracer.start_span(
            name,
            attributes=span_attributes,
            context=trace.set_span_in_context(parent_span) if parent_span else None
        )
        
        return span
    
    def end_span_success(self, span: Span, result: Any = None):
        """End span with success status."""
        span.set_status(Status(StatusCode.OK))
        
        if result is not None:
            span.set_attribute("result.type", type(result).__name__)
        
        span.end()
    
    def end_span_error(self, span: Span, error: Exception):
        """End span with error status."""
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()
    
    def add_baggage(self, key: str, value: str):
        """Add baggage to current context."""
        baggage.set_baggage(key, value)
    
    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from current context."""
        return baggage.get_baggage(key)


class AlertManager:
    """
    Alert management system for system health and issues.
    """
    
    def __init__(self):
        self.alert_handlers = defaultdict(list)
        self.alert_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.logger = logger.bind(component="alert_manager")
    
    def register_alert_handler(
        self,
        severity: AlertSeverity,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """Register an alert handler for specific severity."""
        self.alert_handlers[severity].append(handler)
        self.logger.info(f"Registered alert handler for {severity.value}")
    
    def trigger_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        component: str,
        workflow_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Trigger an alert."""
        alert = {
            "id": str(UUID()),
            "title": title,
            "message": message,
            "severity": severity.value,
            "component": component,
            "workflow_id": str(workflow_id) if workflow_id else None,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add to history
        self.alert_history.append(alert)
        
        # Log the alert
        self.logger.log(
            severity.value.upper(),
            f"ALERT: {title}",
            message=message,
            component=component,
            workflow_id=str(workflow_id) if workflow_id else None,
            **metadata or {}
        )
        
        # Notify handlers
        for handler in self.alert_handlers[severity]:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(
                    "Alert handler failed",
                    handler=handler.__name__,
                    error=str(e)
                )
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return list(self.alert_history)[-limit:]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Dict[str, Any]]:
        """Get alerts by severity level."""
        return [
            alert for alert in self.alert_history 
            if alert["severity"] == severity.value
        ]


class PerformanceMonitor:
    """
    System performance monitoring and analysis.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.performance_data: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.logger = logger.bind(component="performance_monitor")
    
    def record_operation_time(self, operation: str, duration_seconds: float):
        """Record operation execution time."""
        self.performance_data[f"{operation}_duration"].append(duration_seconds)
    
    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size."""
        self.performance_data[f"{queue_name}_queue_size"].append(size)
    
    def record_memory_usage(self, component: str, usage_mb: float):
        """Record memory usage."""
        self.performance_data[f"{component}_memory_mb"].append(usage_mb)
    
    def get_performance_stats(self, metric_name: str) -> Dict[str, float]:
        """Get performance statistics for a metric."""
        data = self.performance_data.get(metric_name, deque())
        
        if not data:
            return {}
        
        data_list = list(data)
        return {
            "count": len(data_list),
            "min": min(data_list),
            "max": max(data_list),
            "avg": sum(data_list) / len(data_list),
            "p95": self._percentile(data_list, 0.95),
            "p99": self._percentile(data_list, 0.99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all performance statistics."""
        return {
            metric_name: self.get_performance_stats(metric_name)
            for metric_name in self.performance_data.keys()
        }


class WorkflowMonitor:
    """
    Specialized monitoring for workflow execution and state changes.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        metrics_collector: MetricsCollector,
        tracing_service: TracingService,
        alert_manager: AlertManager
    ):
        self.state_manager = state_manager
        self.metrics = metrics_collector
        self.tracing = tracing_service
        self.alerts = alert_manager
        self.logger = logger.bind(component="workflow_monitor")
        
        # Monitoring task
        self._monitoring_task = None
    
    async def start_monitoring(self, interval_seconds: float = 30.0):
        """Start workflow monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(
                self._monitoring_loop(interval_seconds)
            )
            self.logger.info("Started workflow monitoring", interval=interval_seconds)
    
    def stop_monitoring(self):
        """Stop workflow monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
            self.logger.info("Stopped workflow monitoring")
    
    async def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop."""
        while True:
            try:
                await self._collect_workflow_metrics()
                await self._check_workflow_health()
                await asyncio.sleep(interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(interval_seconds)
    
    async def _collect_workflow_metrics(self):
        """Collect workflow metrics."""
        try:
            # Count workflows by status
            status_counts = await self._get_workflow_status_counts()
            self.metrics.update_active_workflows(status_counts)
            
            # Count pending approvals by type
            approval_counts = await self._get_pending_approval_counts()
            self.metrics.update_pending_approvals(approval_counts)
            
        except Exception as e:
            self.logger.error("Error collecting workflow metrics", error=str(e))
    
    async def _check_workflow_health(self):
        """Check workflow system health."""
        try:
            # Check for stuck workflows
            stuck_workflows = await self._find_stuck_workflows()
            if stuck_workflows:
                self.alerts.trigger_alert(
                    title="Stuck Workflows Detected",
                    message=f"Found {len(stuck_workflows)} workflows stuck in running state",
                    severity=AlertSeverity.WARNING,
                    component="workflow_engine",
                    metadata={"stuck_count": len(stuck_workflows)}
                )
            
            # Check for overdue approvals
            overdue_approvals = await self._find_overdue_approvals()
            if overdue_approvals:
                self.alerts.trigger_alert(
                    title="Overdue Approvals",
                    message=f"Found {len(overdue_approvals)} overdue approvals",
                    severity=AlertSeverity.WARNING,
                    component="approval_engine",
                    metadata={"overdue_count": len(overdue_approvals)}
                )
            
        except Exception as e:
            self.logger.error("Error checking workflow health", error=str(e))
    
    async def _get_workflow_status_counts(self) -> Dict[str, int]:
        """Get workflow counts by status."""
        # This would query the database for workflow counts
        # Simplified implementation
        workflows = await self.state_manager.list_workflows(limit=1000)
        
        status_counts = defaultdict(int)
        for workflow in workflows:
            status_counts[workflow.status.value] += 1
        
        return dict(status_counts)
    
    async def _get_pending_approval_counts(self) -> Dict[str, int]:
        """Get pending approval counts by type."""
        # This would query the database for pending approvals
        # Simplified implementation - would need approval engine reference
        return {}
    
    async def _find_stuck_workflows(self) -> List[Workflow]:
        """Find workflows that appear to be stuck."""
        # Find workflows running for more than 1 hour
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        stuck_workflows = []
        workflows = await self.state_manager.list_workflows(
            status=WorkflowStatus.RUNNING,
            limit=1000
        )
        
        for workflow in workflows:
            if (workflow.started_at and 
                workflow.started_at < cutoff_time and
                not workflow.updated_at or 
                workflow.updated_at < cutoff_time):
                stuck_workflows.append(workflow)
        
        return stuck_workflows
    
    async def _find_overdue_approvals(self) -> List[Dict[str, Any]]:
        """Find overdue approval requests."""
        # This would query for approvals past their expiration
        # Simplified implementation
        return []
    
    def record_workflow_event(
        self,
        workflow_id: UUID,
        event_type: str,
        duration_seconds: Optional[float] = None
    ):
        """Record workflow event with tracing."""
        span = self.tracing.start_span(
            f"workflow.{event_type}",
            workflow_id=workflow_id,
            attributes={
                "event.type": event_type,
                "workflow.id": str(workflow_id)
            }
        )
        
        if duration_seconds is not None:
            span.set_attribute("duration.seconds", duration_seconds)
        
        self.tracing.end_span_success(span)
    
    def record_approval_event(
        self,
        approval_id: UUID,
        workflow_id: UUID,
        event_type: str,
        approval_type: str,
        channels: List[str]
    ):
        """Record approval event with tracing."""
        span = self.tracing.start_span(
            f"approval.{event_type}",
            workflow_id=workflow_id,
            attributes={
                "approval.id": str(approval_id),
                "approval.type": approval_type,
                "approval.channels": ",".join(channels),
                "event.type": event_type
            }
        )
        
        self.tracing.end_span_success(span)


class ObservabilityManager:
    """
    Central observability manager coordinating all monitoring services.
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        metrics_port: int = 8001,
        jaeger_endpoint: Optional[str] = None
    ):
        self.state_manager = state_manager
        self.logger = logger.bind(component="observability_manager")
        
        # Initialize services
        self.metrics = MetricsCollector()
        self.tracing = TracingService(jaeger_endpoint=jaeger_endpoint)
        self.alerts = AlertManager()
        self.performance = PerformanceMonitor()
        self.workflow_monitor = WorkflowMonitor(
            state_manager, self.metrics, self.tracing, self.alerts
        )
        
        # Start Prometheus metrics server
        self.metrics_port = metrics_port
        self._metrics_server_started = False
    
    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self._metrics_server_started:
            start_http_server(self.metrics_port, registry=self.metrics.registry)
            self._metrics_server_started = True
            self.logger.info("Started metrics server", port=self.metrics_port)
    
    async def start_monitoring(self):
        """Start all monitoring services."""
        self.start_metrics_server()
        await self.workflow_monitor.start_monitoring()
        
        # Register default alert handlers
        self.alerts.register_alert_handler(
            AlertSeverity.CRITICAL,
            self._log_critical_alert
        )
        
        self.logger.info("Started observability monitoring")
    
    async def stop_monitoring(self):
        """Stop all monitoring services."""
        self.workflow_monitor.stop_monitoring()
        self.logger.info("Stopped observability monitoring")
    
    def _log_critical_alert(self, alert: Dict[str, Any]):
        """Log critical alerts."""
        self.logger.critical(
            "CRITICAL ALERT",
            title=alert["title"],
            message=alert["message"],
            component=alert["component"],
            workflow_id=alert["workflow_id"],
            metadata=alert["metadata"]
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        return {
            "metrics": {
                "workflows_active": dict(self.metrics.active_workflows._value),
                "approvals_pending": dict(self.metrics.pending_approvals._value),
            },
            "alerts": {
                "recent_count": len(self.alerts.get_recent_alerts(limit=10)),
                "critical_count": len(self.alerts.get_alerts_by_severity(AlertSeverity.CRITICAL))
            },
            "performance": self.performance.get_all_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }