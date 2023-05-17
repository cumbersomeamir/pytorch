"Installing Libraries"
#!pip3 install torch datasets accelerate transformers tqdm pandas

"Importing Libraries"
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, AdamW , get_scheduler
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset

#Initialising the accelerator object
accelerator = Accelerator()

#Reading the excel file
df = pd.read_excel("Juice Wrld small dataset (3).xlsx")
#Converting Pandas Dataframe to Huggingface Dataset
dataset = Dataset.from_pandas(df)
print("The huggingface dataset is ", dataset)
dataset_dict = dataset.train_test_split(test_size = 0.2)

checkpoint = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return {
        'input_ids': tokenizer(examples["prompt"], truncation=True, padding='max_length', max_length=512)['input_ids'],
        'labels': tokenizer(examples["completion"], truncation=True, padding='max_length', max_length=512)['input_ids'],
    }



tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
print("The type of tokenized datasets is ", tokenized_datasets)
print("The type of tokenized datasets type is ", type(tokenized_datasets))





#Important step - set the format
tokenized_datasets.set_format("torch")
print("The torch format has been set")

#Initialising the DataCollator
data_collator = DataCollatorWithPadding(tokenizer)
print("Data Collator Initialised")

#Defining the train_dataloder and eval_dataloader
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = 8, collate_fn = data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size = 8, collate_fn = data_collator)
print("Train_dataloader and Eval_dataloader have been loaded")

#Loading the Model

model = AutoModel.from_pretrained(checkpoint, num_labels =2)
print("Model has been loaded")


"To check everything is going well, we pass the batch we grabbed to our model. If labels are provided, transformers models always return the loss directly"
#outputs = model(**batch)
#print(output.loss, output.logits.shape)

optimizer = AdamW(model.parameters(), lr = 5e-5)
print("Optimizer has been loaded")

num_epochs = 5
num_training_steps = num_epochs*len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer = optimizer, num_warmup_steps =0, num_training_steps = num_training_steps)


train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)
print("Accelerator prepared")

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


