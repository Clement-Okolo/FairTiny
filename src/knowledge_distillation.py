import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, target, criterion, alpha=0.5, temperature=2.0):
    """
    Apply knowledge distillation loss between student and teacher logits
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        target: Ground truth labels
        criterion: Original loss criterion 
        alpha: Weight of distillation loss vs hard-target loss (0.5 = equal weight)
        temperature: Temperature for softening probabilities
        
    Returns:
        total_loss: Combined loss
        hard_loss: Loss from hard targets
        soft_loss: Loss from soft targets (distillation)
    """
    # Standard supervised loss
    hard_loss = criterion(student_logits, target)
    
    # Knowledge distillation loss
    # Fix for binary classification
    if len(student_logits.shape) == 1 or (len(student_logits.shape) > 1 and student_logits.shape[1] == 1):
        # For binary case with BCEWithLogitsLoss
        # Ensure both student and teacher logits are flattened
        if len(student_logits.shape) > 1:
            student_logits = student_logits.view(-1)
        if len(teacher_logits.shape) > 1:
            teacher_logits = teacher_logits.view(-1)
            
        # Get probabilities instead of logits for KL divergence
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        student_log_probs = F.logsigmoid(student_logits / temperature)
        student_log_one_minus_probs = F.logsigmoid(-student_logits / temperature)
        
        # KL divergence for binary case
        soft_loss = -(teacher_probs * student_log_probs + 
                     (1 - teacher_probs) * student_log_one_minus_probs).mean() * (temperature ** 2)
    else:  # Multi-class classification
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_loss = -(soft_teacher * soft_student).sum(dim=1).mean() * (temperature ** 2)
    
    # Combine losses
    total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
    
    return total_loss, hard_loss, soft_loss
