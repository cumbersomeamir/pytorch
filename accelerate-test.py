"Installing Libraries"
#!pip3 install torch datasets accelerate transformers tqdm

"Importing Libraries"
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, AdamW , get_scheduler
from tqdm.auto import tqdm


accelerator = Accelerator()

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
  return tokenizer(examples["sentence1"], examples["sentence2"], truncation = True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

#Important step - set the format
tokenized_datasets.set_format("torch")

#Initialising the DataCollator
data_collator = DataCollatorWithPadding(tokenizer)

#Defining the train_dataloder and eval_dataloader
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = 8, collate_fn = data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size = 8, collate_fn = data_collator)

#Loading the Model

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels =2)


"To check everything is going well, we pass the batch we grabbed to our model. If labels are provided, transformers models always return the loss directly"
#outputs = model(**batch)
#print(output.loss, output.logits.shape)

optimizer = AdamW(model.parameters(), lr = 5e-5)

num_epochs = 5
num_training_steps = num_epochs*len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer = optimizer, num_warmup_steps =0, num_training_steps = num_training_steps)


train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)


from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range (num_epochs):
  print("The epoch is", epoch)
  for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss #Compute the loss
    accelerator.backward(loss) #Compute Gradients
    optimizer.step() #Optimise
    lr_scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

model.push_to_hub("bert-finetuned")

