import torch
from torch.nn import CrossEntropyLoss


def train_plmr(model,optimizer0, optimizer1, optimizer2, optimizer3, optimizer_dim_reduction, dataset, device, args):
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0
    model.train()
    for (batch, data) in enumerate(dataset):

        optimizer0.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer_dim_reduction.zero_grad()

        inputs = data["input_ids"].to(device)
        masks = data["attention_mask"].to(device)
        labels = data["label"].to(device)

        outputs = model.forward(input_ids=inputs, attention_mask=masks, labels=labels, mode_train=True)


        loss = outputs.loss
        classifier_loss = outputs.classification_loss

        mid_hidden_states = outputs.hidden_states[args.dim_reduction_start + 1]
        mid_cls = mid_hidden_states[:,0,:]
        mid_logits = model.cls_predictor(mid_cls)


        logits = outputs.logits

        loss_fct = CrossEntropyLoss()
        full_text_loss = loss_fct(mid_logits.view(-1, 2), labels.view(-1))

        D_loss = (classifier_loss / args.cls_lambda - full_text_loss)
        loss = loss + full_text_loss * args.full_text_lambda + D_loss * args.D_lambda

        loss.backward()
        optimizer0.step()
        optimizer0.zero_grad()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        optimizer3.step()
        optimizer3.zero_grad()
        optimizer_dim_reduction.step()
        optimizer_dim_reduction.zero_grad()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, predictions = torch.max(cls_soft_logits, dim=-1)

        true_positives += ((predictions == 1) & (labels == 1)).cpu().sum()
        true_negatives += ((predictions == 0) & (labels == 0)).cpu().sum()
        false_negatives += ((predictions == 0) & (labels == 1)).cpu().sum()
        false_positives += ((predictions == 1) & (labels == 0)).cpu().sum()
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    return precision, recall, f1_score, accuracy


def dev_plmr(model, dataset, device):
    """
    Evaluate the PLMR model on a validation or test dataset.

    Parameters:
    - model: The PLMR model to be evaluated.
    - dataset: The evaluation dataset.
    - device: The device to run the model on (CPU or GPU).

    Returns:
    - precision, recall, f1_score, accuracy: Evaluation metrics.
    """
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0

    model.eval()

    with torch.no_grad():
        for batch_index, data in enumerate(dataset):
            inputs = data["input_ids"].to(device)
            attention_masks = data["attention_mask"].to(device)
            labels = data["label"].to(device)

            # Forward pass
            outputs = model.forward(input_ids=inputs, attention_mask=attention_masks, labels=labels, mode_train=False)
            class_probabilities = torch.softmax(outputs.logits, dim=-1)
            _, predictions = torch.max(class_probabilities, dim=-1)

            # Compute metrics
            true_positives += ((predictions == 1) & (labels == 1)).cpu().sum()
            true_negatives += ((predictions == 0) & (labels == 0)).cpu().sum()
            false_negatives += ((predictions == 0) & (labels == 1)).cpu().sum()
            false_positives += ((predictions == 1) & (labels == 0)).cpu().sum()

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return precision, recall, f1_score, accuracy

def annotation_plmr(model, dataset, device, args):
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0
    num_true_positives = 0.0
    num_predicted_positives = 0.0
    num_real_positives = 0.0
    total_words = 0

    model.eval()

    for batch_index, data in enumerate(dataset):

        inputs = data["input_ids"].to(device)
        attention_masks = data["attention_mask"].to(device)
        labels = data["label"].to(device)
        annotations = data["rationale"].to(device)

        # Forward pass
        outputs = model.forward(
            input_ids=inputs, attention_mask=attention_masks, labels=labels, mode_train=False
        )
        logits = outputs.logits

        # Compute predictions
        class_probabilities = torch.softmax(logits, dim=-1)
        _, predictions = torch.max(class_probabilities, dim=-1)

        # Update confusion matrix counts
        true_positives += ((predictions == 1) & (labels == 1)).cpu().sum()
        true_negatives += ((predictions == 0) & (labels == 0)).cpu().sum()
        false_negatives += ((predictions == 0) & (labels == 1)).cpu().sum()
        false_positives += ((predictions == 1) & (labels == 0)).cpu().sum()

        # Extract rationales
        dim_reduction_mask = outputs.dim_reduction_mask
        rationales = dim_reduction_mask[args.dim_reduction_end]
        padded_rationales = torch.zeros(rationales.shape[0], rationales.shape[1], device=rationales.device)
        padded_rationales[:, 1:] = rationales[:, 1:]
        rationales = padded_rationales

        true_pos_count, predicted_pos_count, real_pos_count = rationale_result(annotations, rationales)
        num_true_positives += true_pos_count
        num_predicted_positives += predicted_pos_count
        num_real_positives += real_pos_count
        total_words += torch.sum(attention_masks)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    R_precision = num_true_positives / num_predicted_positives
    R_recall = num_true_positives / num_real_positives
    R_f1 = 2 * (R_precision * R_recall) / (R_precision + R_recall)
    sparsity = num_predicted_positives / total_words

    return precision, recall, f1_score, accuracy, sparsity, R_precision, R_recall, R_f1

def rationale_result(true_labels, predicted_labels):
    num_true_positives = torch.sum(true_labels * predicted_labels)
    num_predicted_positives = torch.sum(predicted_labels)
    num_real_positives = torch.sum(true_labels)

    return num_true_positives, num_predicted_positives, num_real_positives

