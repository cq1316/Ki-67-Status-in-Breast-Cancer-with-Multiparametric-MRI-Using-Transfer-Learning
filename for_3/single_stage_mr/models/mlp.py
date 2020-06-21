import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, num_class, num_feature, num_hidden):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_feature, out_features=num_hidden, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=num_hidden, out_features=num_hidden, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=num_hidden, out_features=num_class, bias=True),
            nn.Softmax()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.1)

    def forward(self, mr_feature):
        out = self.classifier(mr_feature)
        return out
