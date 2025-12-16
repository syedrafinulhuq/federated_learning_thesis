import os
import shutil
import json
import argparse
import time
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report, f1_score
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.model_selection import train_test_split
from src import utils
from src.data_loader import MyDataset, load_data
from src.model import CharacterLevelCNN
from src.focal_loss import FocalLoss
with open('config.json') as f:
    config = json.load(f)

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    threshold = cm_normalized.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_normalized[i, j] > threshold else "black"
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def train(
    model,
    training_generator,
    optimizer,
    criterion,
    epoch,
    writer,
    log_file,
    scheduler,
    class_names,
    args,
    print_every=25,
):
    model.train()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(training_generator)

    progress_bar = tqdm(enumerate(training_generator), total=num_iter_per_epoch)

    y_true = []
    y_pred = []

    for iter, batch in progress_bar:
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        predictions = model(features)

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(predictions, 1)[1].cpu().numpy().tolist()

        loss = criterion(predictions, labels)

        loss.backward()
        if args.scheduler == "clr":
            scheduler.step()

        optimizer.step()
        training_metrics = utils.get_evaluation(
            labels.cpu().numpy(),
            predictions.cpu().detach().numpy(),
            list_metrics=["accuracy", "f1"],
        )

        losses.update(loss.data, features.size(0))
        accuracies.update(training_metrics["accuracy"], features.size(0))

        f1 = training_metrics["f1"]

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + iter)

        writer.add_scalar(
            "Train/Accuracy",
            training_metrics["accuracy"],
            epoch * num_iter_per_epoch + iter,
        )

        writer.add_scalar("Train/f1", f1, epoch * num_iter_per_epoch + iter)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if (iter % print_every == 0) and (iter > 0):
            print(
                "[Training - Epoch: {}], LR: {} , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                    epoch + 1, lr, iter, num_iter_per_epoch, losses.avg, accuracies.avg
                )
            )

            if bool(args.log_f1):
                intermediate_report = classification_report(
                    y_true, y_pred, output_dict=True
                )

                f1_by_class = "F1 Scores by class: "
                for class_name in class_names:
                    f1_by_class += f"{class_name} : {np.round(intermediate_report[class_name]['f1-score'], 4)} |"

                print(f1_by_class)

    f1_train = f1_score(y_true, y_pred, average="weighted")

    writer.add_scalar("Train/loss/epoch", losses.avg, epoch + iter)
    writer.add_scalar("Train/acc/epoch", accuracies.avg, epoch + iter)
    writer.add_scalar("Train/f1/epoch", f1_train, epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, "a") as f:
        f.write(f"Training on Epoch {epoch} \n")
        f.write(f"Average loss: {losses.avg} \n")
        f.write(f"Average accuracy: {accuracies.avg} \n")
        f.write(f"F1 score: {f1_train} \n\n")
        f.write(report)
        f.write("*" * 25)
        f.write("\n")

    return losses.avg, accuracies.avg, f1_train

def evaluate(
    model, validation_generator, criterion, epoch, writer, log_file,class_names, print_every=25
):
    model.eval()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(validation_generator)

    y_true = []
    y_pred = []

    for iter, batch in tqdm(enumerate(validation_generator), total=num_iter_per_epoch):
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            predictions = model(features)
        loss = criterion(predictions, labels)

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(predictions, 1)[1].cpu().numpy().tolist()

        validation_metrics = utils.get_evaluation(
            labels.cpu().numpy(),
            predictions.cpu().detach().numpy(),
            list_metrics=["accuracy", "f1"],
        )
        accuracy = validation_metrics["accuracy"]
        f1 = validation_metrics["f1"]

        losses.update(loss.data, features.size(0))
        accuracies.update(validation_metrics["accuracy"], features.size(0))

        writer.add_scalar("Validation/Loss", loss.item(), epoch * num_iter_per_epoch + iter)
        writer.add_scalar("Validation/Accuracy", accuracy, epoch * num_iter_per_epoch + iter)
        writer.add_scalar("Validation/f1", f1, epoch * num_iter_per_epoch + iter)

        if (iter % print_every == 0) and (iter > 0):
            print(
                "[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                    epoch + 1, iter, num_iter_per_epoch, losses.avg, accuracies.avg
                )
            )

    # Add safety checks for F1 calculation
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"Empty predictions/labels detected! y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        f1_test = 0.0
    else:
        try:
            # Convert to numpy arrays for safety
            y_true_np = np.asarray(y_true)
            y_pred_np = np.asarray(y_pred)
            
            # Handle case where all predictions are same class
            if len(np.unique(y_pred_np)) == 1:
                print("All predictions belong to single class!")
                
            f1_test = f1_score(y_true_np, y_pred_np, average="weighted")
        except UndefinedMetricWarning:
            print("F1 score undefined - setting to 0.0")
            f1_test = 0.0
        except Exception as e:
            print(f"Error calculating F1: {str(e)}")
            f1_test = 0.0

    # Ensure numerical type
    f1_test = float(f1_test) if isinstance(f1_test, (np.generic, float, int)) else 0.0

    # Debug prints
    print(f"Validation samples: {len(y_true)}")
    print(f"Class distribution - True: {Counter(y_true)}")
    print(f"Class distribution - Pred: {Counter(y_pred)}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    fig = plot_confusion_matrix(cm, class_names)  # class_names is your list of label names

    writer.add_figure("Validation/Confusion_Matrix", fig, global_step=epoch)
    plt.close(fig)  # Close the figure after logging to free memory

    writer.add_scalar("Validation/loss/epoch", losses.avg, epoch + iter)
    writer.add_scalar("Validation/acc/epoch", accuracies.avg, epoch + iter)
    writer.add_scalar("Validation/f1/epoch", f1_test, epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report) 

    with open(log_file, "a") as f:
        f.write(f"Validation on Epoch {epoch} \n")
        f.write(f"Average loss: {losses.avg} \n")  # Removed .item()
        f.write(f"Average accuracy: {accuracies.avg} \n")  # Removed .item()
        f.write(f"F1 score {f1_test} \n\n")
        f.write(report)
        f.write("=" * 50)
        f.write("\n")

    return losses.avg, accuracies.avg, f1_test


def run(args, both_cases=False):

    if args.flush_history == 1:
        objects = os.listdir(args.log_path)
        for f in objects:
            if os.path.isdir(args.log_path + f):
                shutil.rmtree(args.log_path + f)

    now = datetime.now()
    logdir = args.log_path + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)
    log_file = logdir + "log.txt"
    writer = SummaryWriter(logdir)

    batch_size = args.batch_size

    training_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }

    validation_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "drop_last": True,
    }

    texts, labels, number_of_classes, sample_weights = load_data(args)

    class_names = sorted(list(set(labels)))
    class_names = [str(class_name) for class_name in class_names]

    # Step 1: Split 70% train, 30% temp (val + test)
    train_texts, temp_texts, train_labels, temp_labels, train_sample_weights, temp_sample_weights = train_test_split(
    texts,
    labels,
    sample_weights,
    test_size=0.30,
    random_state=42,
    stratify=labels
    )

 # Step 2: Split 30% temp into 15% val and 15% test
    val_texts, test_texts, val_labels, test_labels, val_sample_weights, test_sample_weights = train_test_split(
    temp_texts,
    temp_labels,
    temp_sample_weights,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
    )

    # Create datasets
    training_set = MyDataset(train_texts, train_labels, args)
    validation_set = MyDataset(val_texts, val_labels, args)
    # test_set = MyDataset(test_texts, test_labels, args)

    if bool(args.use_sampler):
        train_sample_weights = torch.from_numpy(train_sample_weights)
        sampler = WeightedRandomSampler(
            train_sample_weights.type("torch.DoubleTensor"), len(train_sample_weights)
        )
        training_params["sampler"] = sampler
        training_params["shuffle"] = False

    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    model = CharacterLevelCNN(args, number_of_classes)
    if torch.cuda.is_available():
        model.cuda()

    if not bool(args.focal_loss):
        if bool(args.class_weights):
            class_counts = dict(Counter(train_labels))
            m = max(class_counts.values())
            for c in class_counts:
                class_counts[c] = m / class_counts[c]
            weights = []
            for k in sorted(class_counts.keys()):
                weights.append(class_counts[k])

            weights = torch.Tensor(weights)
            if torch.cuda.is_available():
                weights = weights.cuda()
                print(f"passing weights to CrossEntropyLoss : {weights}")
                criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    else:
        if args.alpha is None:
            criterion = FocalLoss(gamma=args.gamma, alpha=None)
        else:
            criterion = FocalLoss(
                gamma=args.gamma, alpha=[args.alpha] * number_of_classes
            )

    if args.optimizer == "sgd":
        if args.scheduler == "clr":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=1, momentum=0.9, weight_decay=0.00001
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate, momentum=0.9
            )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_f1 = 0
    best_epoch = 0

    if args.scheduler == "clr":
        stepsize = int(args.stepsize * len(training_generator))
        clr = utils.cyclical_lr(stepsize, args.min_lr, args.max_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = None

    for epoch in range(args.epochs):
        print(f"[DEBUG] Starting epoch {epoch}")
        training_loss, training_accuracy, train_f1 = train(
            model,
            training_generator,
            optimizer,
            criterion,
            epoch,
            writer,
            log_file,
            scheduler,
            class_names,
            args,
            args.log_every,
        )
        print(f"[DEBUG] Training done. loss={training_loss}, acc={training_accuracy}, f1={train_f1}")

        validation_loss, validation_accuracy, validation_f1 = evaluate(
            model,
            validation_generator,
            criterion,
            epoch,
            writer,
            log_file,
            class_names,
            args.log_every,
        )
        # os.makedirs(args.output, exist_ok=True)
        # latest_path = os.path.join(args.output, f"latest_epoch_{epoch}.pth")
        # torch.save(model.state_dict(), latest_path)
        # print(f"[INFO] Saved latest model to {latest_path}")
        print(
            "[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}".format(
                epoch + 1,
                args.epochs,
                training_loss,
                training_accuracy,
                validation_loss,
                validation_accuracy,
            )
        )
        print("=" * 50)

        # learning rate scheduling

        if args.scheduler == "step":
            if args.optimizer == "sgd" and ((epoch + 1) % 3 == 0) and epoch > 0:
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                current_lr /= 2
                print("Decreasing learning rate to {0}".format(current_lr))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

        # model checkpoint

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            if args.checkpoint == 1:
                os.makedirs(args.output, exist_ok=True)

                torch.save(
                    model.state_dict(),
                    # os.path.join(args.output, f"best_model_epoch_{epoch}.pth"),
                    args.output
                    + "model_{}_epoch_{}_maxlen_{}_lr_{}_loss_{}_acc_{}_f1_{}.pth".format(
                        args.model_name,
                        epoch,
                        args.max_length,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        round(validation_loss, 4),
                        round(validation_accuracy, 4),
                        round(validation_f1, 4),
                    ),
                )

        if bool(args.early_stopping):
            if epoch - best_epoch > args.patience > 0:
                print(
                    "Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(
                        epoch, validation_loss, best_epoch
                    )
                )
                break
        # === Final Test Evaluation ===
    print("\n[INFO] Running final test set evaluation...\n")

    test_set = MyDataset(test_texts, test_labels, args)
    test_generator = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False
    )

    model.eval()
    test_loss, test_accuracy, test_f1 = evaluate(
        model,
        test_generator,
        criterion,
        epoch=args.epochs,
        writer=writer,
        log_file=log_file,
        class_names=class_names,
        print_every=args.log_every,
    )

    print("=" * 60)
    print(f"[FINAL TEST RESULTS] Accuracy: {test_accuracy:.4f} | F1 Score: {test_f1:.4f}")
    print("=" * 60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Character Based CNN for text classification")
    parser.add_argument("--data_path", type=str, default="./src/url.csv")
    parser.add_argument("--validation_split", type=float, default=0.15)
    parser.add_argument("--label_column", type=str, default="Sentiment")
    parser.add_argument("--text_column", type=str, default="SentimentText")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--steps", nargs="+", default=["lower"])
    parser.add_argument("--group_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ignore_center", type=int, default=1, choices=[0, 1])
    parser.add_argument("--label_ignored", type=int, default=None)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--balance", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_sampler", type=int, default=1, choices=[0, 1])

    parser.add_argument(
        "--alphabet",
        type=str,
        default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",
    )
    parser.add_argument("--number_of_characters", type=int, default=69)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--dropout_input", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--class_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--focal_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.75)

    parser.add_argument(
        "--scheduler", type=str, default="step", choices=["clr", "step"]
    )
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--stepsize", type=float, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stopping", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint", type=int, default=1, choices=[0, 1])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_f1", type=int, default=1, choices=[0, 1])
    parser.add_argument("--flush_history", type=int, default=1, choices=[0, 1])
    parser.add_argument("--output", type=str, default="./models/")
    parser.add_argument("--model_name", type=str, default="")

    args = parser.parse_args()
    os.makedirs(args.log_path, exist_ok=True)
    print(f"Running with arguments: {args}")
    run(args)
