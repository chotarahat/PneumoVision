from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import build_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_index is None:
            class_index = logits.argmax(dim=1).item()

        score = logits[:, class_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        cam = cam.squeeze().cpu()
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)

        probability = torch.softmax(logits, dim=1)[0, class_index].item()
        return cam.numpy(), class_index, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="models/pneumovision_resnet18.pth")
    parser.add_argument("--output", default="reports/figures/gradcam.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]

    model = build_model(
        num_classes=len(classes),
        pretrained=False,
        dropout=checkpoint["config"]["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(args.image).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model, model.layer4[-1].conv2)
    heatmap, class_index, probability = cam(tensor)

    original = image.resize((224, 224))

    plt.figure(figsize=(7, 7))
    plt.imshow(original, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(f"{classes[class_index]} ({probability:.2%})")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
