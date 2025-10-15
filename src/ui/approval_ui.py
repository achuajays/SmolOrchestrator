"""
Dynamic approval UI system that renders configurable approval forms
from metadata/schemas with multi-framework support.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import streamlit as st
import gradio as gr
from pydantic import BaseModel, Field

from ..models.workflow import ApprovalRequest, ChannelType
from ..core.approval_engine import ApprovalEngine


class ApprovalUIComponent(BaseModel):
    """Base class for approval UI components."""
    
    component_type: str
    title: str
    description: Optional[str] = None
    required: bool = False
    default_value: Any = None
    validation_rules: Optional[Dict[str, Any]] = None


class StreamlitApprovalUI:
    """
    Streamlit-based dynamic approval UI renderer.
    """
    
    def __init__(self, approval_engine: ApprovalEngine):
        self.approval_engine = approval_engine
    
    def render_approval_form(
        self,
        approval: ApprovalRequest,
        approver_id: str
    ) -> Optional[Dict[str, Any]]:
        """Render approval form using Streamlit components."""
        
        st.markdown(f"# {approval.title}")
        st.markdown(f"**Description:** {approval.description}")
        
        # Show context information
        if approval.context_data:
            with st.expander("Additional Context", expanded=False):
                st.json(approval.context_data)
        
        # Show expiration info
        time_left = approval.expires_at - datetime.now(timezone.utc)
        if time_left.total_seconds() > 0:
            hours_left = time_left.total_seconds() / 3600
            st.info(f"⏰ This approval expires in {hours_left:.1f} hours")
        else:
            st.error("⚠️ This approval has expired")
            return None
        
        # Render form based on UI schema
        ui_schema = approval.ui_schema or {}
        form_data = {}
        
        with st.form(key=f"approval_form_{approval.id}"):
            # Render form fields based on schema
            form_data = self._render_schema_fields(ui_schema)
            
            # Add submission buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                approved = st.form_submit_button("✅ Approve", type="primary")
            with col2:
                rejected = st.form_submit_button("❌ Reject", type="secondary")
            with col3:
                if approval.approval_type in ["form", "budget"]:
                    modified = st.form_submit_button("🔄 Request Changes")
                else:
                    modified = False
            
            # Process submission
            if approved:
                form_data["decision"] = "approved"
                return self._submit_approval(approval.id, approver_id, form_data)
            elif rejected:
                form_data["decision"] = "rejected"
                return self._submit_approval(approval.id, approver_id, form_data)
            elif modified:
                form_data["decision"] = "modified"
                return self._submit_approval(approval.id, approver_id, form_data)
        
        return None
    
    def _render_schema_fields(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Render form fields based on JSON schema."""
        form_data = {}
        properties = schema.get("properties", {})
        ui_config = schema.get("ui", {})
        required_fields = schema.get("required", [])
        
        for field_name, field_schema in properties.items():
            if field_name == "decision":
                continue  # Skip decision field as it's handled by buttons
            
            field_type = field_schema.get("type", "string")
            field_title = field_schema.get("title", field_name.title())
            field_description = field_schema.get("description", "")
            is_required = field_name in required_fields
            default_value = field_schema.get("default")
            
            # Add required indicator
            if is_required:
                field_title += " *"
            
            # Render field based on type
            if field_type == "string":
                if field_schema.get("format") == "textarea":
                    form_data[field_name] = st.text_area(
                        field_title,
                        value=default_value or "",
                        help=field_description,
                        height=ui_config.get(field_name, {}).get("ui:options", {}).get("rows", 3) * 25
                    )
                elif field_schema.get("enum"):
                    options = field_schema["enum"]
                    option_labels = field_schema.get("enumNames", options)
                    form_data[field_name] = st.selectbox(
                        field_title,
                        options=options,
                        format_func=lambda x: option_labels[options.index(x)] if x in options else x,
                        index=options.index(default_value) if default_value in options else 0,
                        help=field_description
                    )
                else:
                    form_data[field_name] = st.text_input(
                        field_title,
                        value=default_value or "",
                        help=field_description
                    )
            
            elif field_type == "number":
                min_val = field_schema.get("minimum", 0.0)
                max_val = field_schema.get("maximum", 1000000.0)
                form_data[field_name] = st.number_input(
                    field_title,
                    min_value=min_val,
                    max_value=max_val,
                    value=float(default_value) if default_value is not None else min_val,
                    help=field_description
                )
            
            elif field_type == "integer":
                min_val = field_schema.get("minimum", 0)
                max_val = field_schema.get("maximum", 1000000)
                form_data[field_name] = st.number_input(
                    field_title,
                    min_value=min_val,
                    max_value=max_val,
                    value=int(default_value) if default_value is not None else min_val,
                    step=1,
                    help=field_description
                )
            
            elif field_type == "boolean":
                form_data[field_name] = st.checkbox(
                    field_title,
                    value=bool(default_value) if default_value is not None else False,
                    help=field_description
                )
            
            elif field_type == "array":
                # Handle arrays as multiselect
                items_enum = field_schema.get("items", {}).get("enum", [])
                if items_enum:
                    form_data[field_name] = st.multiselect(
                        field_title,
                        options=items_enum,
                        default=default_value if isinstance(default_value, list) else [],
                        help=field_description
                    )
                else:
                    # Simple text area for array input
                    text_input = st.text_area(
                        field_title,
                        value="\n".join(default_value) if isinstance(default_value, list) else "",
                        help=field_description + " (Enter one item per line)"
                    )
                    form_data[field_name] = [line.strip() for line in text_input.split("\n") if line.strip()]
        
        return form_data
    
    async def _submit_approval(
        self,
        approval_id: UUID,
        approver_id: str,
        form_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit approval response."""
        try:
            response = await self.approval_engine.respond_to_approval(
                approval_id=approval_id,
                approver_id=approver_id,
                decision=form_data["decision"],
                response_data=form_data,
                feedback=form_data.get("feedback", ""),
                channel_type=ChannelType.UI
            )
            
            st.success(f"✅ Approval {form_data['decision']} successfully submitted!")
            return {"success": True, "response_id": str(response.id)}
            
        except Exception as e:
            st.error(f"❌ Error submitting approval: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def render_approval_list(self, approver_id: str) -> None:
        """Render list of pending approvals for an approver."""
        st.markdown("# Pending Approvals")
        
        # Get pending approvals
        pending_approvals = await self.approval_engine.list_pending_approvals(
            requested_from=approver_id
        )
        
        if not pending_approvals:
            st.info("No pending approvals found.")
            return
        
        # Display approvals in expandable cards
        for approval in pending_approvals:
            time_left = approval.expires_at - datetime.now(timezone.utc)
            hours_left = time_left.total_seconds() / 3600
            
            # Determine urgency color
            if hours_left < 1:
                urgency_color = "red"
                urgency_icon = "🔴"
            elif hours_left < 6:
                urgency_color = "orange"
                urgency_icon = "🟡"
            else:
                urgency_color = "green"
                urgency_icon = "🟢"
            
            with st.expander(
                f"{urgency_icon} {approval.title} - {hours_left:.1f}h left",
                expanded=(hours_left < 6)
            ):
                st.markdown(f"**Description:** {approval.description}")
                st.markdown(f"**Priority:** {'High' if approval.priority > 5 else 'Normal' if approval.priority > 0 else 'Low'}")
                st.markdown(f"**Requested by:** {approval.requested_by}")
                st.markdown(f"**Created:** {approval.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
                
                if st.button(f"Review Approval", key=f"review_{approval.id}"):
                    st.session_state.selected_approval = str(approval.id)
                    st.rerun()


class GradioApprovalUI:
    """
    Gradio-based approval UI for web interfaces.
    """
    
    def __init__(self, approval_engine: ApprovalEngine):
        self.approval_engine = approval_engine
    
    def create_approval_interface(self, approval_id: str) -> gr.Interface:
        """Create a Gradio interface for a specific approval."""
        
        async def process_approval(
            decision: str,
            feedback: str,
            approver_id: str,
            *additional_fields
        ):
            """Process approval submission."""
            try:
                # Get approval details
                approval = await self.approval_engine.get_approval(UUID(approval_id))
                if not approval:
                    return "Error: Approval not found"
                
                # Build response data
                form_data = {
                    "decision": decision,
                    "feedback": feedback
                }
                
                # Add additional fields based on schema
                if approval.ui_schema and "properties" in approval.ui_schema:
                    field_names = [
                        name for name in approval.ui_schema["properties"].keys()
                        if name not in ["decision", "feedback"]
                    ]
                    
                    for i, field_value in enumerate(additional_fields):
                        if i < len(field_names):
                            form_data[field_names[i]] = field_value
                
                # Submit approval
                response = await self.approval_engine.respond_to_approval(
                    approval_id=UUID(approval_id),
                    approver_id=approver_id,
                    decision=decision,
                    response_data=form_data,
                    feedback=feedback,
                    channel_type=ChannelType.UI
                )
                
                return f"✅ Approval {decision} successfully submitted!"
                
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        # Create interface components
        inputs = [
            gr.Radio(
                choices=["approved", "rejected"],
                label="Decision",
                value="approved"
            ),
            gr.Textbox(
                label="Feedback",
                placeholder="Optional feedback...",
                lines=3
            ),
            gr.Textbox(
                label="Approver ID",
                placeholder="Enter your user ID"
            )
        ]
        
        outputs = gr.Textbox(label="Result")
        
        interface = gr.Interface(
            fn=process_approval,
            inputs=inputs,
            outputs=outputs,
            title=f"Approval Request",
            description="Review and respond to approval request",
            theme="default"
        )
        
        return interface
    
    def create_approval_dashboard(self, approver_id: str) -> gr.Interface:
        """Create approval dashboard interface."""
        
        async def get_pending_approvals():
            """Get pending approvals for display."""
            approvals = await self.approval_engine.list_pending_approvals(
                requested_from=approver_id
            )
            
            if not approvals:
                return "No pending approvals"
            
            result = []
            for approval in approvals:
                time_left = approval.expires_at - datetime.now(timezone.utc)
                hours_left = time_left.total_seconds() / 3600
                
                result.append(f"""
                **{approval.title}**
                - Description: {approval.description}
                - Time left: {hours_left:.1f} hours
                - Priority: {approval.priority}
                - ID: {approval.id}
                ---
                """)
            
            return "\n".join(result)
        
        interface = gr.Interface(
            fn=get_pending_approvals,
            inputs=[],
            outputs=gr.Markdown(),
            title="Approval Dashboard",
            description="View your pending approvals"
        )
        
        return interface


class ApprovalUIRenderer:
    """
    Universal approval UI renderer that can work with multiple UI frameworks.
    """
    
    def __init__(self, approval_engine: ApprovalEngine):
        self.approval_engine = approval_engine
        self.streamlit_ui = StreamlitApprovalUI(approval_engine)
        self.gradio_ui = GradioApprovalUI(approval_engine)
    
    def render_approval_email_html(self, approval: ApprovalRequest) -> str:
        """Generate HTML for email-based approvals."""
        
        # Base URL for approval endpoints (would be configured)
        base_url = "https://your-approval-system.com"
        
        context_html = ""
        if approval.context_data:
            context_html = f"""
            <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-radius: 5px;">
                <h4>Additional Context:</h4>
                <pre style="white-space: pre-wrap;">{json.dumps(approval.context_data, indent=2)}</pre>
            </div>
            """
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Approval Required: {approval.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #007bff; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #fff; padding: 30px; border: 1px solid #ddd; }}
                .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 0 0 5px 5px; }}
                .btn {{ display: inline-block; padding: 12px 24px; margin: 10px; text-decoration: none; border-radius: 5px; font-weight: bold; }}
                .btn-approve {{ background-color: #28a745; color: white; }}
                .btn-reject {{ background-color: #dc3545; color: white; }}
                .btn-review {{ background-color: #6c757d; color: white; }}
                .urgent {{ border-left: 5px solid #ff6b6b; padding-left: 15px; }}
                .normal {{ border-left: 5px solid #4ecdc4; padding-left: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔔 Approval Required</h1>
                    <h2>{approval.title}</h2>
                </div>
                
                <div class="content">
                    <div class="{'urgent' if approval.priority > 5 else 'normal'}">
                        <h3>Description:</h3>
                        <p>{approval.description}</p>
                        
                        {context_html}
                        
                        <h4>Details:</h4>
                        <ul>
                            <li><strong>Requested by:</strong> {approval.requested_by}</li>
                            <li><strong>Priority:</strong> {'High' if approval.priority > 5 else 'Normal' if approval.priority > 0 else 'Low'}</li>
                            <li><strong>Expires:</strong> {approval.expires_at.strftime('%Y-%m-%d %H:%M UTC')}</li>
                            <li><strong>Type:</strong> {approval.approval_type.title()}</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <h3>Quick Actions:</h3>
                        <a href="{base_url}/approve/{approval.id}?decision=approved" class="btn btn-approve">
                            ✅ Approve
                        </a>
                        <a href="{base_url}/approve/{approval.id}?decision=rejected" class="btn btn-reject">
                            ❌ Reject
                        </a>
                        <a href="{base_url}/approval/{approval.id}" class="btn btn-review">
                            📋 Detailed Review
                        </a>
                    </div>
                </div>
                
                <div class="footer">
                    <p><small>
                        This approval request was generated automatically. 
                        If you cannot use the buttons above, visit: {base_url}/approval/{approval.id}
                    </small></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def generate_slack_blocks(self, approval: ApprovalRequest) -> List[Dict[str, Any]]:
        """Generate Slack Block Kit components for approval requests."""
        
        # Determine urgency
        time_left = approval.expires_at - datetime.now(timezone.utc)
        hours_left = time_left.total_seconds() / 3600
        
        if hours_left < 1:
            urgency_emoji = "🔴"
            urgency_text = "URGENT"
        elif hours_left < 6:
            urgency_emoji = "🟡"
            urgency_text = "HIGH"
        else:
            urgency_emoji = "🟢"
            urgency_text = "NORMAL"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{urgency_emoji} Approval Required: {approval.title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{approval.description}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Requested by:*\n{approval.requested_by}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Priority:*\n{urgency_text}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time left:*\n{hours_left:.1f} hours"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Type:*\n{approval.approval_type.title()}"
                    }
                ]
            }
        ]
        
        # Add context data if present
        if approval.context_data:
            context_text = json.dumps(approval.context_data, indent=2)
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Additional Context:*\n```{context_text}```"
                }
            })
        
        # Add action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Approve"
                    },
                    "style": "primary",
                    "value": f"approve_{approval.id}",
                    "action_id": "approval_approve"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "❌ Reject"
                    },
                    "style": "danger",
                    "value": f"reject_{approval.id}",
                    "action_id": "approval_reject"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📋 Review Details"
                    },
                    "value": f"review_{approval.id}",
                    "action_id": "approval_review",
                    "url": f"https://your-approval-system.com/approval/{approval.id}"
                }
            ]
        })
        
        return blocks