import wandb
import torch
import matplotlib.pyplot as plt

wandb.init(project="ts2vec", entity="arjunashok", tags=["plot"])

gt = torch.randn(512,)
pred = torch.randn(512,)

gt = gt.cpu().numpy()
pred = pred.cpu().numpy()

# plot lines
plt.plot(list(range(gt.shape[0])), gt, label = "gt")
plt.plot(list(range(gt.shape[0])), pred, label = "pred")

plt.legend()
# plt.show()

wandb.log({"forecast_plots/1_hour_ahead": plt})

wandb.finish()