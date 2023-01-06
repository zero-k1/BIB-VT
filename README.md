
# Using this Project  
  
## Getting started  
  
Set your ```PYTHONPATH``` to the repository:  
  
``` export PYTHONPATH="${PYTHONPATH}:/path/to/repo" ```  
  
Install all necessary python packages with a package manager:  
  
``` pip install -r requirements.txt ```  
  
## Data preparation  
  
#### Download the BIB dataset  
To retrieve the BIB dataset, go to the [BIB website](https://www.kanishkgandhi.com/bib) and download the data splits.  
  
#### Preprocessing  
Because the json files are very large and contain more information than needed (e.g., the shape and color of agents and goals), we preprocessed the json files in the BIB datasets to only retain necessary information. This can be done by running the ``preprocess.py`` script in ```data``` as follows:  
  
``python preprocess.py json path/to/input/folder path/to/output/folder``, e.g.  
  
``python preprocess.py json /Downloads/bib-training/multi_agent /Projects/BIB-VT/data/bib_train/multi_agent``  
  
You can also omit this step, but you will need to adapt the ``datasets.py`` file accordingly. We also recommend to preprocess the video clips, so that you don't have to load the whole dataset anew every time you use it:  
  
``python preprocess.py vid path/to/input/folder path/to/output/folder``, e.g.  
  
``python preprocess.py vid multi_agent train``  
  
Train, test, val, and eval sets need to be preprocessed separately for each task! 
  
In the end, your data folder should look something like this:  
  
```  
data  
└───bib_train  
│   │   index_dict_test_preference.json  
│   │   index_dict_train_preference.json  
│   │   index_dict_val_preference.json  
│   │   index_dict_test_multi_agent.json  
│   │   ...  
│   └── preference  
│       │   000000pre.json  
│       │   000000pre.mp4  
│       │   ...  
│ └───bib_eval  
│   │   index_dict_eval_preference.json  
│   │   index_dict_eval_multi_agent.json  
│   │   ...  
│   └── preference  
│   └── multi_agent  
```  
where the .json files are your *preprocessed* .json files (if using).  
  
## Settings  
  
You can change the training and model parameters via the ``settings.py`` file, where you can also specify the path to your background training and evaluation data (``data_path_train``, ``data_path_eval``), where to save your trained models (``model_dir``), and what to name your model when saving (``model_name``). The ``saved_models`` directory contains some pre-trained models.  
  
## Training  
Once you have finalized your settings, you can train a model using ``train.py``:  
  
``python train.py``  
  
## Evaluating  
  
**IMPORTANT**: For VoE evaluation on the BIB tasks, set the batch size in ````settings.py```` to ``2``!  
  
To e.g. evaluate model_4.pt on the "preference" task, use:   
  
``python eval.py eval preference model_4.pt``  
  
If you want to use the loss of the most surprising frames as the evaluation metric (rather than the mean error over the whole trial), set the ``reduction="none"`` option for ``nn.BCEWithLogitsLoss`` in ``eval.py``.  
  
# Acknowledgments  
This code builds in parts on other repositories, namely:  
  
* Kanishk Gandhi's [bib-baselines](https://github.com/kanishkg/bib-baselines) repository (MIT License), for loading the data  
* Phil Wang's [vit-pytorch](https://github.com/lucidrains/vit-pytorch) repository (MIT License), for parts of the transformer model code
