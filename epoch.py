import random
import time
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, model, train_dataloader, val_dataloader, optimizer, loss_fn):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(args.epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        # t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            if args.model_state == 'BERT_classifier':
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                model.zero_grad()

                logits = model(b_input_ids, b_attn_mask)
            
            else:
                b_input_ids, b_labels = tuple(t.to(device) for t in batch)
                model.zero_grad()
                logits = model(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            # scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                # time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                # t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        
        # =======================================
        #               Evaluation
        # =======================================
        
        # After the completion of each training epoch, measure the model's performance
        # on our validation set.
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn)
        # Print performance over the entire training data
        # time_elapsed = time.time() - t0_epoch
        
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f}")
        print("-"*70)
    print("\n")
    
    print("Training complete!")


def evaluate(args, model, test_dataloader, loss_fn):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    print("Start evaluate...\n")
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        # Load batch to GPU
        if args.model_state == 'BERT_classifier':
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                logits = model(b_input_ids, b_attn_mask)

                with torch.no_grad():
                    logits = model(b_input_ids, b_attn_mask)
            
        else:
            b_input_ids, b_labels = tuple(t for t in batch)
            
            with torch.no_grad():
                    logits = model(b_input_ids)
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    
    return val_loss, val_accuracy