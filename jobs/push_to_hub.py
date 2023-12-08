from transformers import AutoModel

model = AutoModel.from_pretrained("train/models/distillation_converted/dist_1")
print("Starting push to hub")
model.push_to_hub("Nicolas-BZRD/pythia-410-squad-distillation_1")
print("Push to hub done")