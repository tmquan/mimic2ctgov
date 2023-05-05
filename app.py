from huggingface_hub import snapshot_download
from txtai.workflow import Workflow
from txtai.workflow import Task
from txtai.pipeline import Tabular
from txtai.pipeline import Similarity
from txtai.embeddings import Embeddings
import json
import pandas as pd
import gradio as gr
import numpy as np

import os
os.environ['TRANSFORMERS_CACHE'] = 'cache'

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x): return x


class SemanticSearch(object):
    def __init__(
        self,
        filename="ctgov",
        columns=[
            "brief_title",
            "official_title",
            "brief_summaries",
            "detailed_descriptions",
            "criteria",
        ],
        ckptlist=[
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        ],
        rerun=True,
    ):
        self.filename = filename
        self.ckptlist = ckptlist

        for ckptpath in self.ckptlist:
            snapshot_download(repo_id=ckptpath,
                              repo_type="model",
                              cache_dir="cache")
            self.embeddings = Embeddings({
                "method": "transformers",
                "path": ckptpath,
                "content": True,
                "object": True
            })
            indexfile = f'{filename}_{ckptpath.replace("/", "-")}.index'
            if os.path.exists(indexfile) and rerun is False:
                print("Indexed and Cached!")
                self.embeddings.load(indexfile)
            else:
                print("Need to rerun or Indices and Caches dont exist, run them!")

                # Create tabular instance mapping input.csv fields
                tabular = Tabular(idcolumn="nct_id", textcolumns=columns, content=True)

                # Create workflow
                workflow = Workflow([Task(tabular)])

                # Index subset of CORD-19 data
                data = list(workflow([f'{filename}.csv']))
                # print(data[:1])
                self.embeddings.index(data)
                self.embeddings.save(indexfile)
                print("Indexing and Caching finished for the 1st time!")
    
    def func(self, 
             file_paths, 
             pretrained="sentence-transformers/multi-qa-mpnet-base-dot-v1", 
             limit=10):
        #
        # Parsing the EHR first
        #
        dfs = []
        # [print(file_path.name) for file_path in file_paths]
        for file_path in file_paths:
            if file_path.name.endswith('.json'):
                df = pd.read_json(file_path.name)
                
            elif file_path.name.endswith('.csv'):
                df = pd.read_csv(file_path.name)
            else:
                print(f"Unsupported file format: {file_path}")
                continue
            df.reset_index()
            dfs.append(df)
        dfs = pd.concat(dfs)
        # print(dfs)
        dfs = dfs.reset_index(drop=True) #.set_index("subject_id")
        json_ehr = dfs.to_json(orient='records')
        
        tempfile = "temp_ehr.csv"
        dfs.to_csv(tempfile)
        
        #
        # searching for relevant clinical trials
        #
        # Create tabular instance mapping input.csv fields
        tabular = Tabular(idcolumn="subject_id", textcolumns=["description", "text"], content=True)
        # Create workflow
        workflow = Workflow([Task(tabular)])

        queries = list(workflow([f"{tempfile}"]))
        queries = [str(query) for query in queries]
        results = [self.embeddings.search(query, limit) for query in queries]
        
        return json_ehr, results
    
    def launch_interface(self, *args, **kwargs):
        interface = gr.Interface(
            fn=lambda *args, **kwargs: self.func(*args, **kwargs),
            inputs=[
                gr.File(file_count="multiple"),
                gr.Dropdown(
                    self.ckptlist, value="sentence-transformers/multi-qa-mpnet-base-dot-v1", label=f"Pretrained model"),
                gr.Slider(1, 20, value=5, label=f"Number of similar trials", step=1),
            ],
            outputs=[
                gr.Json(label=f"Parsed EHR data"),
                gr.Json(label=f"Similar trials if found")
            ],
            # examples=[
            #     ["hypertension", "sentence-transformers/multi-qa-mpnet-base-dot-v1", 3],
            #     ["diabetes", "sentence-transformers/multi-qa-mpnet-base-dot-v1", 2],
            # ],
            title="Semantic Search on Clinical Trial Data",
            description="Uploading bulk of clinical trials",
        ) # type: ignore
        interface.launch(*args, **kwargs)


def main():
    trial_search = SemanticSearch(
        filename="ctgov",
        columns=[
            "brief_title",
            "official_title",
            "brief_summaries",
            "detailed_descriptions",
            "criteria",
        ],
        ckptlist=[
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        ],
        rerun=False,
    )
    trial_search.launch_interface(server_name="localhost", 
                                  share=True)


if __name__ == "__main__":
    main()
