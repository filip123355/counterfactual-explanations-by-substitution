from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt

from src.constants import CLASSIFIER_LABEL, CLASSIFIER_MODEL_PATH, USE_HQ


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DenseNet121(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.densenet121(weights=None)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


ID_TO_CLS = {
    "celeb": [
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Attractive",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ],
    "celebhq": ["male", "smiling", "young"],
}


class DenseNetClassifier(torch.nn.Module):
    def __init__(self, path_ckpt: str, dataset_name: Literal["celeb", "celebhq"]):
        super().__init__()

        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(
            self.feat_extract.output_size, len(ID_TO_CLS[dataset_name])
        )
        self.transforms = tt.Compose(
            [tt.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
        self.label_id = list(map(lambda s: s.lower(), ID_TO_CLS[dataset_name])).index(
            CLASSIFIER_LABEL.lower()
        )

        self.load_ckpt(path_ckpt)
        self.eval()

    def use_functional_relu_only(self):
        pass

    def load_ckpt(self, path_ckpt):
        ckpt = torch.load(path_ckpt, map_location="cpu")

        if "feat_extract" in ckpt:
            self.feat_extract.load_state_dict(ckpt["feat_extract"])
        if "classifier" in ckpt:
            self.classifier.load_state_dict(ckpt["classifier"])
        if "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])

    def forward(self, x):
        # NOTE: Input is required to be in [0, 1] range
        x = self.transforms(x)
        x = self.feat_extract(x)
        x = self.classifier(x)

        # every multilabel classifier can be converted to 'multiclass' classifier by:
        #   1. constraining the predictions to include only the label of interest
        #   2. modifying its output to provide pairs of (p, 1 - p), where p denotes
        #       the probability for the class of interest.

        if self.label_id is not None:
            # pick column correspondig to label of interest
            x = x[:, [self.label_id]]

            # add column with its repeated logits
            x = x.repeat(1, 2)

            # make it negative to later represent 1 - p, where p
            # is the probability determined by initial logits
            x[:, 1] = -x[:, 1]

        return x

    def pred_prob(self, x):
        x = self(x)
        return F.sigmoid(x)

    def pred_label(self, x):
        x = self.pred_prob(x)
        return x.argmax(dim=1).long()


def get_classifier():
    clf = DenseNetClassifier(
        CLASSIFIER_MODEL_PATH, dataset_name="celebhq" if USE_HQ else "celeb"
    )
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False
    return clf
