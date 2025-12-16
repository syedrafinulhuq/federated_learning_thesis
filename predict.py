import argparse
import torch
import torch.nn.functional as F
from src.model import CharacterLevelCNN
from src import utils
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()


def predict(args):
    model = CharacterLevelCNN(args, args.number_of_classes)
    state = torch.load(args.model)
    model.load_state_dict(state)
    model.eval()

    processed_input = utils.preprocess_input(args)
    processed_input = torch.tensor(processed_input)
    processed_input = processed_input.unsqueeze(0)
    if use_cuda:
        processed_input = processed_input.to("cuda")
        model = model.to("cuda")
    prediction = model(processed_input)
    probabilities = F.softmax(prediction, dim=1)
    probabilities = probabilities.detach().cpu().numpy()
    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str,default="models/model__epoch_1_maxlen_150_lr_0.01_loss_0.0248_acc_0.8549_f1_0.9038.pth")
    parser.add_argument("--text", type=str, default="http://example.com/adult-content", help="URL to classify as Adult or Not Adult")
    parser.add_argument("--steps", nargs="+", default=["lower"])

    # arguments needed for the predicition
    parser.add_argument(
        "--alphabet",
        type=str,
        default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",
    )
    parser.add_argument("--number_of_characters", type=int, default=69)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--number_of_classes", type=int, default=2)
    parser.add_argument("--dropout_input", type=float, default=0.3)


    args = parser.parse_args()
    prediction = predict(args)

    print("input : {}".format(args.text))
    print("prediction : {}".format(prediction))
    labels = ["Not Adult", "Adult"]
    plt.bar(labels, prediction[0])
    plt.title(f"Prediction for URL: {args.text}")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.show()