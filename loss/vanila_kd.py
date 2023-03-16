from loss.contrastive import loss_function as Contrastive
from loss.softmax import loss_function as LabelDriven

class loss_function:
    def __init__(self):
        self.a = 0.5

    def __call__(self, y_hat, y_teacher, target):
        contrastive_loss = Contrastive(y_hat, y_teacher)
        ce_loss = LabelDriven(y_hat, target)
        return self.a * contrastive_loss + (1-self.a) * ce_loss