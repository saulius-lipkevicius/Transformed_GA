import torch.onnx
from torch import nn
#importuoti modeli 


if __name__=="__main__":
    device = "cuda"
    review = ["This is script to convert BERT model to ONNX"]

    dataset = CustomeDataset(
        Dataset = input_dataset #sulyginti su class cusotmedatasets
    )

    model = Model(bert_model)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    model.eval()

    print(model)

    #defining model inputs, which are ids, kmask and toke type ids (it comes from Model Class)
    input_ids= dataset[0]['token_ids'].unsqueeze(0)
    attn_masks = dataset[0]['attn_masks'].unsqueeze(0)
    token_type_ids = dataset[0]['token_type_ids'].unsqueeze(0)

    torch.onnx.export(
        model, #.module if paralized
        (input_ids, attn_masks, token_type_ids),
        "model.onnx",
        input_names=['input_ids', 'attn_masks', 'token_type_ids'], 
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0, "batch_size"},
            "attn_masks": {0, "batch_size"},
            "token_type_ids": {0, "batch_size"},
            "output": {0, "batch_size"},
        }, #which inputs have dynamical axes
    )





