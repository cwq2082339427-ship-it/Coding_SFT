import json
from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset


class CodeAlpaca(Dataset):
    def __init__(self,path):
        self.data = load_dataset(path)["train"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        item = self.data[index]
        prompt = item["prompt"]
        output = item["completion"]

        message = [
            {"role":"user","content" : prompt},
            {"role":"assistant","content" : output}
        ]

        return {
            "messages": message
        }
    
class OpcSftStage2(Dataset):
    def __init__(self,path):
        self.data = load_dataset(path,name="educational_instruct")["train"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        item = self.data[index]

        instruction = item["instruction"]
        output = item["code"]

        message = [
            {"role":"user","content" :instruction},
            {"role":"assistant","content":output}
        ]
        return {
            "messages":message
        }
    
class EvolInstructCode80k(Dataset):
    def __init__(self,path):
        self.data = load_dataset(path)["train"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        item = self.data[index]

        instruction = item["instruction"]
        output = item["output"]

        message = [
            {"role":"user","content" :instruction},
            {"role":"assistant","content":output}
        ]

        return {
            "messages":message
        }
    
dataset1 = CodeAlpaca("HuggingFaceH4/CodeAlpaca_20K")
dataset2 = OpcSftStage2("OpenCoder-LLM/opc-sft-stage2")
dataset3 = EvolInstructCode80k("nickrosh/Evol-Instruct-Code-80k-v1")

mydataset_without_clean = ConcatDataset(
    [dataset1,dataset2,dataset3]
)
print(mydataset_without_clean[0])