import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
# This is for the progress bar.
from tqdm.auto import tqdm


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            # 卷积层过后，向量shape变为64*128*128
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 最大池化层过后，向量shape变为64*64*64
            nn.MaxPool2d(2, 2, 0),
            # 卷积层过后，向量shape变为128*64*64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 最大池化层过后，向量shape变为128*32*32
            nn.MaxPool2d(2, 2, 0),
            # 卷积层过后，向量shape变为256*32*32
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 最大池化层过后，向量shape变为256*8*8
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            # 128/2/2/4=8
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        # For the classification task, we use cross-entropy as the measurement of performance.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        # 256*8*8
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

    def cal_loss(self, outputs, labels):
        return self.criterion(outputs, labels)


def get_pseudo_labels(dataloader, model, device, threshold=0.65):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    # Initialize lists to store pseudo-labeled data.
    pseudo_images = []
    pseudo_labels = []
    # Iterate over the dataset by batches.
    for batch in tqdm(dataloader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
        # Iterate over each prediction probability.
        for prob in probs:
            # Check if the maximum probability exceeds the threshold.
            if prob.max() > threshold:
                # Get the index of the maximum probability as the predicted label.
                pred_label = prob.argmax().item()
                # Append the image and its predicted label to the pseudo-labeled lists.
                pseudo_images.append(img)
                pseudo_labels.append(pred_label)

    # Convert the pseudo-labeled data to tensors.
    pseudo_images = torch.cat(pseudo_images, dim=0)
    pseudo_labels = torch.tensor(pseudo_labels)
    # Create a new dataset using the pseudo-labeled data.
    pseudo_dataset = torch.utils.data.TensorDataset(pseudo_images, pseudo_labels)
    # # Turn off the eval mode.
    model.train()
    return pseudo_dataset


def train_val(model, config, train_set, train_loader, unlabeled_loader, valid_loader, device):
    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    # Whether to do semi-supervised learning.
    do_semi = True
    for epoch in range(config["n_epochs"]):
        # ---------- TODO ----------
        # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
        # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
        if do_semi:
            # Obtain pseudo-labels for unlabeled data using trained model.
            pseudo_set = get_pseudo_labels(unlabeled_loader, model, device)
            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = model.cal_loss(logits, labels.to(device))
            # Compute the gradients for parameters.
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            # Update the parameters with computed gradients.
            optimizer.step()
            # Compute the accuracy for current batch.
            # _, train_pred = torch.max(outputs, 1)不仅得到了值，还得到了索引
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):
                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                logits = model(imgs.to(device))
                # We can still compute the loss (but not the gradient).
                loss = model.cal_loss(logits, labels.to(device))
                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


def test(model, test_loader, device):
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()
    # Initialize a list to store the predictions.
    predictions = []
    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        # Iterate the testing set by batches.
        for batch in tqdm(test_loader):
            # A batch consists of image data and corresponding labels.
            # But here the variable "labels" is useless since we do not have the ground-truth.
            # If printing out the labels, you will find that it is always 0.
            # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
            # so we have to create fake labels to make it work normally.
            imgs, labels = batch
            logits = model(imgs.to(device))
            # Take the class with greatest logit as prediction and record it.
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return predictions