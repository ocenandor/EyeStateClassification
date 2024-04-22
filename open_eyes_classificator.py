import torch
from PIL import Image
from torchvision import transforms as T

from src.model import ResNet


class OpenEyesClassificator:
    def __init__(self) -> None:
        self.model = ResNet()
        self.model.load_state_dict(torch.load('model_weights/best_model.ckpt')['state_dict'])
        self.model.eval()
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((24, 24)),
            ])

    def predict(self, inpIm) -> float:
        img = Image.open(inpIm)
        img = self.transforms(img).unsqueeze(0)
        with torch.no_grad():
            is_open_score = self.model(img)[0][1].item()
        return is_open_score 