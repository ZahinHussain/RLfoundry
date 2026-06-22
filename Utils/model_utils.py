import torch


def save_model(self, path):
    checkpoint = {
        "policy_net": self.policy_net.state_dict(),
        "target_net": self.target_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "frame": getattr(self, "frame", None),
        "reward_history": getattr(self, "reward_history", None),
    }
    torch.save(checkpoint, path)


def load_model(self, path):
    checkpoint = torch.load(path, map_location=self.device)

    self.policy_net.load_state_dict(checkpoint["policy_net"])
    self.target_net.load_state_dict(checkpoint["target_net"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])

    if "reward_history" in checkpoint and checkpoint["reward_history"] is not None:
        self.reward_history = checkpoint["reward_history"]

    print(f"Loaded model from {path}")