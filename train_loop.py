# Training loop
def train_loop(model, optimizer, loss_fn, anchors, positives, negatives, batch_size):
    optimizer.zero_grad()

    total_loss = 0.0
    num_batches = len(anchors) // batch_size

    if num_batches == 0:
        print("Dataset size is smaller than the batch size.")
        return 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        anchor_sentences = anchors[start_idx:end_idx]
        positive_sentences = positives[start_idx:end_idx]
        negative_sentences = negatives[start_idx:end_idx]

        anchor_embeddings = model.get_embeddings(anchor_sentences)
        positive_embeddings = model.get_embeddings(positive_sentences)
        negative_embeddings = model.get_embeddings(negative_sentences)

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
        # loss = Variable(loss, requires_grad=True)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / num_batches
    return average_loss
