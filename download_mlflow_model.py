import mlflow
import torch

run_id = '90cf0bced379454aa5b2f58410f845a0'
destination_path = './models'


downloaded_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    dst_path=destination_path
)

print(f"Artifacts from run {run_id} downloaded to: {downloaded_path}")

model_uri = f"runs:/{run_id}/model"

torch_model = mlflow.pytorch.load_model(model_uri)

torch.save(torch_model.state_dict(), destination_path + '/model.pth')

print(f"PyTorch model saved locally to: {destination_path}")