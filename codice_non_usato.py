# non usato
def save_model_coeffs(m, filename):
    torch.save(m.state_dict(), filename)
    print("Saved model coefficients to disk")

# non usato
def load_model_coeffs(m, filename):
    m.load_state_dict(torch.load(filename))
    print("Model coefficients loaded from disk")

# non usato
def accuracy(preds, targets):  #@save
    """Compute the number of correct predictions."""
    # deal with the case when an array of probabilities is predict, by deriving the highest-probability class
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        preds = preds.argmax(axis=1)
    cmp = preds.type(targets.dtype) == targets
    return float(cmp.type(targets.dtype).sum())

# non usato
def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        # Set the model to eval mode
        net.eval()
    h_test = History(['correct_predictions', 'predictions'])  # No. of correct predictions, no. of predictions
    with torch.no_grad():  # Gradients must not be computed
        count = 0
        for X, y in data_iter:
            count += 1
            if count % 10 == 0:
                print('x', end='')
            #X = X.flatten(start_dim=1, end_dim=-1)
            h_test.add(accuracy(net(X), y), len(y))
        s = h_test.sums()
    print(' ')
    return s['correct_predictions'] / s['predictions']

# non usato
def train_epoch(model, train_iter, loss_func, optimizer):
    if isinstance(model, torch.nn.Module):
        model.train()  # Set the model to training mode
    h_epoch = History(
        ['loss', 'correct_predictions', 'n_examples'])  # Training loss, no. of correct predictions, no. of examples
    count = 0
    for X, y in train_iter:
        count += 1
        if count % 10 == 0:
            print('.', end='')
        #X=X.flatten(start_dim=1, end_dim=-1)
        # Compute predictions
        y_hat = model(X)
        # Compute loss
        loss = loss_func(y_hat, y)
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        h_epoch.add(float(loss), accuracy(y_hat, y), len(y))
    # Return training loss and training accuracy
    s = h_epoch.sums()
    return s['loss'] / s['n_examples'], s['correct_predictions'] / s['n_examples'], h_epoch

# non usato
def train(net, loaders, loss_func, num_epochs, updater, report=False):
    h_batch = History(['loss', 'correct_predictions', 'n_examples'])
    h_train = History(['training_loss', 'training_accuracy',
                       'test_accuracy'])  # Avg. training loss, avg. training accuracy, test accuracy
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch + 1}')
        # train model for one epoch
        train_loss, train_acc, h_epoch = train_epoch(net, loaders['train'], loss_func, updater)
        # evaluate accuracy on test set
        test_acc = evaluate_accuracy(net, loaders['test'])
        if report:
            print(f' Loss {train_loss:3.4f}, Training set accuracy {train_acc:1.4f}, Test set accuracy {test_acc:1.4f}')
        else:
            print('\n')
        h_train.add(train_loss, train_acc, test_acc)
        h_batch.merge(h_epoch.data)
    return h_train, h_batch

# non usato
def predict(net, loaders):
    preds_train = []
    y_train = []
    preds_test = []
    y_test = []
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        for X, y in loaders['train']:
            #X = X.flatten(start_dim=1, end_dim=-1)
            preds = (torch.max(net(X), 1)[1]).numpy()
            preds_train.extend(preds)
            y_train.extend(y.numpy())
        for X, y in loaders['test']:
            #X = X.flatten(start_dim=1, end_dim=-1)
            preds = (torch.max(net(X), 1)[1]).numpy()
            preds_test.extend(preds)
            y_test.extend(y.numpy())
    return preds_train, y_train, preds_test, y_test

# non usato
def plot_metrics(title, h, hb):
    losses = []
    accs = []
    for l, c, n in zip(hb.data['loss'], hb.data['correct_predictions'], hb.data['n_examples']):
        losses.append(l / n)
        accs.append(c / n)
    n = len(accs)
    step = int(n / num_epochs)
    xs = range(step - 1, n + step - 1, step)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(range(len(losses)), losses, color=colors[0], lw=.5, label='Loss by batch', zorder=2)
    ax1.scatter(xs, h.data['training_loss'], color=colors[1], edgecolors='black', label='Loss by epoch', zorder=2)
    for x in xs:
        ax1.axvline(x, lw=1, color='gray', zorder=1)
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(range(len(accs)), accs, color=colors[0], lw=.5, label='Training set accuracy by batch', zorder=2)
    ax2.scatter(xs, h.data['training_accuracy'], color=colors[1], edgecolors='black',
                label='Training set accuracy by epoch', zorder=2)
    ax2.scatter(xs, h.data['test_accuracy'], color=colors[2], edgecolors='black', label='Test set accuracy by epoch',
                zorder=2)
    for x in xs:
        ax2.axvline(x, lw=1, color='gray', zorder=1)
    ax2.legend()
    ax2.set_title('Accuracy')
    plt.suptitle(title)
    plt.show()

# non usato
def plot_label_dist(predictions_probs, predicted_class, true_label):
    plt.figure(figsize=(4, 4))
    thisplot = plt.bar(range(10), predictions_probs, color="#77aaaa")
    plt.ylim([0, 1])
    thisplot[predicted_class].set_color(colors[0])
    thisplot[true_label].set_edgecolor(colors[4])
    thisplot[true_label].set_linewidth(1)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plt.show()