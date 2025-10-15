"""
Multi-channel integration system for sending and receiving approvals
through Slack, email, and other communication channels.
"""

import asyncio
import json
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID

import structlog
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from pydantic import BaseModel

from ..models.workflow import ApprovalRequest, ChannelType
from ..core.approval_engine import ApprovalEngine
from ..ui.approval_ui import ApprovalUIRenderer

logger = structlog.get_logger(__name__)


class ChannelConfig(BaseModel):
    """Base configuration for communication channels."""
    
    channel_type: ChannelType
    name: str
    enabled: bool = True
    config: Dict[str, Any] = {}


class SlackChannelHandler:
    """
    Slack integration for sending approval requests and handling responses.
    """
    
    def __init__(
        self,
        bot_token: str,
        signing_secret: str,
        approval_engine: ApprovalEngine,
        ui_renderer: ApprovalUIRenderer
    ):
        self.approval_engine = approval_engine
        self.ui_renderer = ui_renderer
        self.logger = logger.bind(component="slack_handler")
        
        # Initialize Slack app
        self.app = AsyncApp(
            token=bot_token,
            signing_secret=signing_secret
        )
        self.client = AsyncWebClient(token=bot_token)
        
        # Register event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register Slack event handlers."""
        
        @self.app.action("approval_approve")
        async def handle_approve_action(ack, body, say):
            await ack()
            await self._handle_approval_action(body, "approved", say)
        
        @self.app.action("approval_reject")
        async def handle_reject_action(ack, body, say):
            await ack()
            await self._handle_approval_action(body, "rejected", say)
        
        @self.app.action("approval_review")
        async def handle_review_action(ack, body, say):
            await ack()
            await self._handle_review_action(body, say)
        
        @self.app.command("/approvals")
        async def handle_approvals_command(ack, body, say):
            await ack()
            await self._handle_list_approvals(body, say)
        
        @self.app.shortcut("approval_shortcut")
        async def handle_approval_shortcut(ack, body, say):
            await ack()
            await self._handle_approval_shortcut(body, say)
    
    async def send_approval_request(self, approval: ApprovalRequest) -> bool:
        """Send approval request via Slack."""
        try:
            # Generate Slack blocks
            blocks = self.ui_renderer.generate_slack_blocks(approval)
            
            # Get channel or user to send to
            target = self._get_slack_target(approval)
            if not target:
                self.logger.error(
                    "No Slack target found for approval",
                    approval_id=str(approval.id)
                )
                return False
            
            # Send message
            result = await self.client.chat_postMessage(
                channel=target,
                text=f"Approval Required: {approval.title}",
                blocks=blocks
            )
            
            # Store message info for future updates
            message_ts = result["ts"]
            self._store_message_info(approval.id, target, message_ts)
            
            self.logger.info(
                "Approval request sent via Slack",
                approval_id=str(approval.id),
                channel=target,
                message_ts=message_ts
            )
            
            return True
            
        except SlackApiError as e:
            self.logger.error(
                "Failed to send Slack approval request",
                approval_id=str(approval.id),
                error=e.response["error"]
            )
            return False
        except Exception as e:
            self.logger.error(
                "Unexpected error sending Slack approval",
                approval_id=str(approval.id),
                error=str(e)
            )
            return False
    
    async def _handle_approval_action(
        self,
        body: Dict[str, Any],
        decision: str,
        say: Callable
    ):
        """Handle approval button clicks."""
        try:
            # Extract approval ID from button value
            action_value = body["actions"][0]["value"]
            approval_id_str = action_value.split("_")[1]
            approval_id = UUID(approval_id_str)
            
            # Get user info
            user_id = body["user"]["id"]
            user_info = await self.client.users_info(user=user_id)
            approver_id = user_info["user"]["profile"]["email"]
            
            # Submit approval
            response = await self.approval_engine.respond_to_approval(
                approval_id=approval_id,
                approver_id=approver_id,
                decision=decision,
                response_data={"decision": decision},
                feedback="Response via Slack",
                channel_type=ChannelType.SLACK,
                channel_metadata={
                    "user_id": user_id,
                    "channel": body.get("channel", {}).get("id"),
                    "message_ts": body.get("message", {}).get("ts")
                }
            )
            
            # Update the original message
            await self._update_approval_message(
                body.get("channel", {}).get("id"),
                body.get("message", {}).get("ts"),
                approval_id,
                decision,
                approver_id
            )
            
            # Send confirmation
            await say(
                text=f"✅ Approval {decision} by {approver_id}",
                thread_ts=body.get("message", {}).get("ts")
            )
            
        except Exception as e:
            self.logger.error(
                "Error handling Slack approval action",
                error=str(e),
                body=body
            )
            await say(f"❌ Error processing approval: {str(e)}")
    
    async def _handle_review_action(self, body: Dict[str, Any], say: Callable):
        """Handle review button clicks."""
        try:
            action_value = body["actions"][0]["value"]
            approval_id_str = action_value.split("_")[1]
            
            # The URL should be handled by the button itself
            await say(
                text="Opening detailed review page...",
                thread_ts=body.get("message", {}).get("ts")
            )
            
        except Exception as e:
            self.logger.error("Error handling review action", error=str(e))
    
    async def _handle_list_approvals(self, body: Dict[str, Any], say: Callable):
        """Handle /approvals command."""
        try:
            user_id = body["user_id"]
            user_info = await self.client.users_info(user=user_id)
            approver_id = user_info["user"]["profile"]["email"]
            
            # Get pending approvals
            approvals = await self.approval_engine.list_pending_approvals(
                requested_from=approver_id,
                limit=10
            )
            
            if not approvals:
                await say("No pending approvals found.")
                return
            
            # Build response
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📋 Your Pending Approvals ({len(approvals)})"
                    }
                }
            ]
            
            for approval in approvals[:5]:  # Show top 5
                time_left = approval.expires_at - datetime.now(timezone.utc)
                hours_left = time_left.total_seconds() / 3600
                
                urgency = "🔴" if hours_left < 1 else "🟡" if hours_left < 6 else "🟢"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{urgency} *{approval.title}*\n{approval.description[:100]}..."
                    },
                    "accessory": {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Review"
                        },
                        "url": f"https://your-approval-system.com/approval/{approval.id}",
                        "action_id": "open_review"
                    }
                })
            
            await say(blocks=blocks)
            
        except Exception as e:
            self.logger.error("Error listing approvals", error=str(e))
            await say(f"❌ Error retrieving approvals: {str(e)}")
    
    async def _handle_approval_shortcut(self, body: Dict[str, Any], say: Callable):
        """Handle approval shortcuts."""
        # This would open a modal for creating new approval requests
        pass
    
    def _get_slack_target(self, approval: ApprovalRequest) -> Optional[str]:
        """Get Slack channel or user ID to send the approval to."""
        # This would typically look up user/channel mapping
        # For now, return a default channel or user DM
        channel_configs = approval.channel_configs or {}
        slack_config = channel_configs.get("slack", {})
        
        return slack_config.get("channel") or slack_config.get("user")
    
    def _store_message_info(self, approval_id: UUID, channel: str, message_ts: str):
        """Store message information for future updates."""
        # This would typically store in database or cache
        pass
    
    async def _update_approval_message(
        self,
        channel: str,
        message_ts: str,
        approval_id: UUID,
        decision: str,
        approver: str
    ):
        """Update the original approval message after response."""
        try:
            updated_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✅ This approval has been *{decision}* by {approver}"
                    }
                }
            ]
            
            await self.client.chat_update(
                channel=channel,
                ts=message_ts,
                blocks=updated_blocks,
                text=f"Approval {decision} by {approver}"
            )
            
        except Exception as e:
            self.logger.error("Error updating Slack message", error=str(e))


class EmailChannelHandler:
    """
    Email integration for sending approval requests and handling responses.
    """
    
    def __init__(
        self,
        sendgrid_api_key: Optional[str] = None,
        smtp_config: Optional[Dict[str, Any]] = None,
        approval_engine: ApprovalEngine = None,
        ui_renderer: ApprovalUIRenderer = None
    ):
        self.sendgrid_client = None
        if sendgrid_api_key:
            self.sendgrid_client = SendGridAPIClient(api_key=sendgrid_api_key)
        
        self.smtp_config = smtp_config
        self.approval_engine = approval_engine
        self.ui_renderer = ui_renderer
        self.logger = logger.bind(component="email_handler")
    
    async def send_approval_request(self, approval: ApprovalRequest) -> bool:
        """Send approval request via email."""
        try:
            # Get email addresses
            recipients = self._get_email_recipients(approval)
            if not recipients:
                self.logger.error(
                    "No email recipients found for approval",
                    approval_id=str(approval.id)
                )
                return False
            
            # Generate email content
            subject = f"Approval Required: {approval.title}"
            html_content = self.ui_renderer.render_approval_email_html(approval)
            text_content = self._generate_text_content(approval)
            
            # Send via SendGrid or SMTP
            if self.sendgrid_client:
                success = await self._send_via_sendgrid(
                    recipients, subject, html_content, text_content
                )
            elif self.smtp_config:
                success = await self._send_via_smtp(
                    recipients, subject, html_content, text_content
                )
            else:
                self.logger.error("No email configuration available")
                return False
            
            if success:
                self.logger.info(
                    "Approval request sent via email",
                    approval_id=str(approval.id),
                    recipients=recipients
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Error sending email approval request",
                approval_id=str(approval.id),
                error=str(e)
            )
            return False
    
    async def _send_via_sendgrid(
        self,
        recipients: List[str],
        subject: str,
        html_content: str,
        text_content: str
    ) -> bool:
        """Send email via SendGrid."""
        try:
            from_email = Email(self.smtp_config.get("from_email", "noreply@company.com"))
            
            # Create message
            mail = Mail(
                from_email=from_email,
                to_emails=[To(email) for email in recipients],
                subject=subject,
                html_content=Content("text/html", html_content),
                plain_text_content=Content("text/plain", text_content)
            )
            
            # Send
            response = self.sendgrid_client.send(mail)
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            self.logger.error("SendGrid send error", error=str(e))
            return False
    
    async def _send_via_smtp(
        self,
        recipients: List[str],
        subject: str,
        html_content: str,
        text_content: str
    ) -> bool:
        """Send email via SMTP."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(recipients)
            
            # Add content parts
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            html_part = MIMEText(html_content, 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send via SMTP
            server = smtplib.SMTP(
                self.smtp_config['host'],
                self.smtp_config.get('port', 587)
            )
            
            if self.smtp_config.get('use_tls', True):
                server.starttls()
            
            if self.smtp_config.get('username'):
                server.login(
                    self.smtp_config['username'],
                    self.smtp_config['password']
                )
            
            server.send_message(msg, to_addrs=recipients)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error("SMTP send error", error=str(e))
            return False
    
    def _get_email_recipients(self, approval: ApprovalRequest) -> List[str]:
        """Get email recipients for the approval."""
        channel_configs = approval.channel_configs or {}
        email_config = channel_configs.get("email", {})
        
        recipients = []
        
        # Add primary recipient
        if approval.requested_from:
            recipients.append(approval.requested_from)
        
        # Add additional recipients from config
        additional = email_config.get("recipients", [])
        recipients.extend(additional)
        
        return list(set(recipients))  # Remove duplicates
    
    def _generate_text_content(self, approval: ApprovalRequest) -> str:
        """Generate plain text email content."""
        time_left = approval.expires_at - datetime.now(timezone.utc)
        hours_left = time_left.total_seconds() / 3600
        
        content = f"""
APPROVAL REQUIRED: {approval.title}

Description:
{approval.description}

Details:
- Requested by: {approval.requested_by}
- Priority: {'High' if approval.priority > 5 else 'Normal' if approval.priority > 0 else 'Low'}
- Expires: {approval.expires_at.strftime('%Y-%m-%d %H:%M UTC')} ({hours_left:.1f} hours left)
- Type: {approval.approval_type.title()}

To respond to this approval, please visit:
https://your-approval-system.com/approval/{approval.id}

Quick actions:
- Approve: https://your-approval-system.com/approve/{approval.id}?decision=approved
- Reject: https://your-approval-system.com/approve/{approval.id}?decision=rejected

This is an automated message. Please do not reply to this email.
        """.strip()
        
        return content


class WebhookChannelHandler:
    """
    Webhook integration for sending approval notifications to external systems.
    """
    
    def __init__(self, approval_engine: ApprovalEngine):
        self.approval_engine = approval_engine
        self.logger = logger.bind(component="webhook_handler")
    
    async def send_approval_request(self, approval: ApprovalRequest) -> bool:
        """Send approval request via webhook."""
        try:
            import aiohttp
            
            # Get webhook configuration
            channel_configs = approval.channel_configs or {}
            webhook_config = channel_configs.get("webhook", {})
            
            webhook_url = webhook_config.get("url")
            if not webhook_url:
                self.logger.error(
                    "No webhook URL configured for approval",
                    approval_id=str(approval.id)
                )
                return False
            
            # Prepare payload
            payload = {
                "approval_id": str(approval.id),
                "workflow_id": str(approval.workflow_id),
                "title": approval.title,
                "description": approval.description,
                "approval_type": approval.approval_type,
                "requested_from": approval.requested_from,
                "requested_by": approval.requested_by,
                "priority": approval.priority,
                "expires_at": approval.expires_at.isoformat(),
                "context_data": approval.context_data,
                "ui_schema": approval.ui_schema,
                "callback_url": f"https://your-approval-system.com/api/approvals/{approval.id}/respond"
            }
            
            # Add custom headers if specified
            headers = webhook_config.get("headers", {})
            headers.setdefault("Content-Type", "application/json")
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status in [200, 201, 202]:
                        self.logger.info(
                            "Webhook sent successfully",
                            approval_id=str(approval.id),
                            webhook_url=webhook_url,
                            status=response.status
                        )
                        return True
                    else:
                        self.logger.error(
                            "Webhook failed",
                            approval_id=str(approval.id),
                            webhook_url=webhook_url,
                            status=response.status
                        )
                        return False
            
        except Exception as e:
            self.logger.error(
                "Error sending webhook",
                approval_id=str(approval.id),
                error=str(e)
            )
            return False


class ChannelManager:
    """
    Manages all communication channels for the approval system.
    """
    
    def __init__(self, approval_engine: ApprovalEngine, ui_renderer: ApprovalUIRenderer):
        self.approval_engine = approval_engine
        self.ui_renderer = ui_renderer
        self.handlers = {}
        self.logger = logger.bind(component="channel_manager")
    
    def register_slack_handler(
        self,
        bot_token: str,
        signing_secret: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Register Slack channel handler."""
        handler = SlackChannelHandler(
            bot_token=bot_token,
            signing_secret=signing_secret,
            approval_engine=self.approval_engine,
            ui_renderer=self.ui_renderer
        )
        self.handlers[ChannelType.SLACK] = handler
        self.logger.info("Slack handler registered")
    
    def register_email_handler(
        self,
        sendgrid_api_key: Optional[str] = None,
        smtp_config: Optional[Dict[str, Any]] = None
    ):
        """Register email channel handler."""
        handler = EmailChannelHandler(
            sendgrid_api_key=sendgrid_api_key,
            smtp_config=smtp_config,
            approval_engine=self.approval_engine,
            ui_renderer=self.ui_renderer
        )
        self.handlers[ChannelType.EMAIL] = handler
        self.logger.info("Email handler registered")
    
    def register_webhook_handler(self):
        """Register webhook channel handler."""
        handler = WebhookChannelHandler(self.approval_engine)
        self.handlers[ChannelType.WEBHOOK] = handler
        self.logger.info("Webhook handler registered")
    
    async def send_approval_notification(
        self,
        approval: ApprovalRequest,
        channels: Optional[List[ChannelType]] = None
    ) -> Dict[ChannelType, bool]:
        """Send approval notification through specified channels."""
        if channels is None:
            channels = [ChannelType(ch) for ch in approval.channels]
        
        results = {}
        
        for channel_type in channels:
            if channel_type in self.handlers:
                try:
                    success = await self.handlers[channel_type].send_approval_request(approval)
                    results[channel_type] = success
                    
                    self.logger.info(
                        "Channel notification sent",
                        approval_id=str(approval.id),
                        channel=channel_type.value,
                        success=success
                    )
                    
                except Exception as e:
                    results[channel_type] = False
                    self.logger.error(
                        "Channel notification failed",
                        approval_id=str(approval.id),
                        channel=channel_type.value,
                        error=str(e)
                    )
            else:
                results[channel_type] = False
                self.logger.warning(
                    "No handler registered for channel",
                    channel=channel_type.value,
                    approval_id=str(approval.id)
                )
        
        return results
    
    def get_slack_app(self) -> Optional[AsyncApp]:
        """Get Slack app instance for web server integration."""
        slack_handler = self.handlers.get(ChannelType.SLACK)
        return slack_handler.app if slack_handler else None