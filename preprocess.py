import json
import os
import sys
from bib_vt.settings import data_path_eval, data_path_train, device
from bib_data.datasets import FrameDataset

def preprocess_jsons(input_dir, output_dir):
    directory = os.fsencode(input_dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            to_save = {}
            with open(input_dir + "/" + filename) as json_file:
                print(filename)
                data = json.load(json_file)
                for i, trial in enumerate(data):
                    to_save[i] = {}
                    for j, frame in enumerate(trial):
                        to_save[i][j] = frame['agent'][0]

            with open(output_dir + "/" + filename, 'w') as outfile:
              json.dump(to_save, outfile)

if __name__ == "__main__":
    if sys.argv[1] == "json":
        input_dir_str =  sys.argv[2]
        output_dir_str =  sys.argv[3]
        preprocess_jsons(input_dir_str, output_dir_str)
    elif sys.argv[1] == "vid":
        if sys.argv[3] == "eval":
            dataset =FrameDataset(data_path_eval, types=[sys.argv[2]], mode=sys.argv[3], process_data=1,
                                  device=device)
        else:
            dataset = FrameDataset(data_path_train, types=[sys.argv[2]], mode=sys.argv[3], process_data=1,
                                        device=device)
    else:
        print("First argument should be 'json' or 'vid'")