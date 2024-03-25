import torch 
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import pipeline

torch.set_grad_enabled(False)

MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)


fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

# Using the mask token correctly
results = fill_mask("Nagyon szeretem reggelire a kenyeret és a tejet.  Almát viszont nem szeretek [MASK], mert nagyon savanyú.")

# Print the results
for result in results:
    print(result)