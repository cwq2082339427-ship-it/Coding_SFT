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



class MyDataset(Dataset):
    def __init__(self, path_list=None):
        if path_list is None:
            path_list = ["HuggingFaceH4/CodeAlpaca_20K", "OpenCoder-LLM/opc-sft-stage2","nickrosh/Evol-Instruct-Code-80k-v1"] 
        self.datasets = [
            CodeAlpaca(path_list[0]),
            OpcSftStage2(path_list[1]),
            EvolInstructCode80k(path_list[2])
        ]
        self.combined = ConcatDataset(self.datasets)
    
    def __len__(self):
        return len(self.combined)
    
    def __getitem__(self, idx):
        return self.combined[idx]

