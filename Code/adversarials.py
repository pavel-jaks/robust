import torch
import torch.nn as nn

class ClassificationAdversarials:
    @staticmethod
    def get_adversarials(
        model: nn.Module,
        benign_examples: torch.Tensor,
        labels: torch.Tensor,
        adversarial_examples: torch.Tensor
    ):
        possible_adversarials = []
        benign_predictions = model(benign_examples)
        adversarial_predictions = model(adversarial_examples)
        benign_classes, benign_confidences = torch.argmax(benign_predictions, dim=1), torch.max(benign_predictions, dim=1)
        adversarial_classes, adversarial_confidences = torch.argmax(adversarial_predictions, dim=1), torch.max(adversarial_predictions, dim=1)
        for i in range(len(labels)):
            benign_class, benign_confidence = int(benign_classes[i]), float(benign_confidences[0][i])
            adversarial_class, adversarial_confidence = int(adversarial_classes[i]), float(adversarial_confidences[0][i])
            if adversarial_class != labels[i] and benign_class == labels[i]:
                params = {"Label": labels[i], "Prediction": adversarial_class, "Confidence": adversarial_confidence,
                        "Index": i, "OriginalPrediction": benign_class, "OriginalConfidence": benign_confidence}
                possible_adversarials.append(params)
        return possible_adversarials
