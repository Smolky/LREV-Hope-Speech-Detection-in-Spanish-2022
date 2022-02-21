import torch
from transformers import BertTokenizer, VisualBertModel

model = VisualBertModel.from_pretrained ("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained ("bert-base-uncased")

image_path = '/home/rafa_pepe/deeplearningsuite/assets/mami/2021/images/1.jpg'

inputs = tokenizer ("What is the man eating?", return_tensors="pt")
# this is a custom function that returns the visual embeddings given the image path
visual_embeds = get_visual_embeddings (image_path)

visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

print ("---------------------")
print (last_hidden_state)

print ("---------------------")
print (visual_embeds)