import torch




def save_model(self, path):
    checkpoint = {
        "policy_net": self.policy_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "reward_history": getattr(self, "reward_history", None),
    }
    torch.save(checkpoint, path)


