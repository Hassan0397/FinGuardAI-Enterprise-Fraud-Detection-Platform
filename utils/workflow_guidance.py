# utils/workflow_guidance.py
from typing import Dict, Any, List

class WorkflowGuidance:
    """Provides workflow guidance and recommendations"""
    
    def __init__(self):
        self.workflow_steps = {
            'upload': {'order': 1, 'name': 'Upload Data', 'icon': '📤', 'required': True},
            'validation': {'order': 2, 'name': 'Validate Data', 'icon': '🔍', 'required': True},
            'detection': {'order': 3, 'name': 'Fraud Detection', 'icon': '🤖', 'required': True},
            'decision': {'order': 4, 'name': 'Decision Engine', 'icon': '⚖️', 'required': False},
            'explain': {'order': 5, 'name': 'Explainability', 'icon': '🧠', 'required': False},
            'analytics': {'order': 6, 'name': 'Analytics', 'icon': '📊', 'required': False}
        }
    
    def get_next_step(self, current_step: str, completed_steps: List[str]) -> Dict[str, Any]:
        """Get the next recommended step in workflow"""
        
        current_order = self.workflow_steps.get(current_step, {}).get('order', 0)
        
        for step, info in self.workflow_steps.items():
            if info['order'] > current_order and step not in completed_steps:
                return {'step': step, 'info': info}
        
        return None
    
    def get_workflow_status(self, completed_steps: List[str]) -> Dict[str, Any]:
        """Get overall workflow status"""
        
        total_steps = len(self.workflow_steps)
        completed_count = len([s for s in completed_steps if s in self.workflow_steps])
        
        return {
            'progress': (completed_count / total_steps) * 100,
            'completed': completed_count,
            'total': total_steps,
            'next_steps': self.get_pending_steps(completed_steps)
        }
    
    def get_pending_steps(self, completed_steps: List[str]) -> List[str]:
        """Get list of pending steps"""
        return [step for step in self.workflow_steps.keys() if step not in completed_steps]
    
    def get_recommendation(self, current_step: str, has_data: bool, has_predictions: bool) -> str:
        """Get workflow recommendation based on current state"""
        
        if not has_data:
            return "📤 Please upload transaction data to begin"
        
        if current_step == 'upload' and has_data:
            return "🔍 Proceed to data validation"
        
        if current_step == 'validation' and has_data:
            return "🤖 Run fraud detection models"
        
        if current_step == 'detection' and has_predictions:
            return "⚖️ Apply decision intelligence engine"
        
        if current_step == 'decision' and has_predictions:
            return "🧠 View explainable AI insights"
        
        return "✅ Workflow in progress"