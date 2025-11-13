import torch
from model.model import CNNLSTMModel
from model.preprocess import scale_image

def load_model(checkpoint_path, device='cuda'):
    model = CNNLSTMModel(img_height=224, img_width=160)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Modèle chargé depuis {checkpoint_path}")
    return model


@torch.no_grad()
def infer_single_snapshot(model, snapshot, hidden=None, device='cuda'):
    img_tensor = torch.tensor(scale_image(snapshot.image)).unsqueeze(0).to(device)

    if hidden is None:
        hidden = model.init_hidden(batch_size=1, device=device)
    
    pred_raycasts, pred_speed, pred_class, hidden = model.forward_step(img_tensor, hidden)
    pred_controls = torch.sigmoid(pred_class).cpu().numpy().flatten()
    
    return pred_controls, hidden