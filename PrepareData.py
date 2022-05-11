import os
import numpy as np
import argparse
import time
import DataLoadingUtils.PrepareData
import DataLoadingUtils.LoadKUMulti

def main():
    parser = argparse.ArgumentParser()
    parser = DataLoadingUtils.PrepareData.get_args(parser)
    parser.add_argument("--data_type", default='vhdr', choices=['vhdr', 'MAT'],
                         help="Whether to consider raw vhdr or preprocessed MAT files [vhdr, MAT]")
    args = parser.parse_args()
    load_data = DataLoadingUtils.LoadKUMulti.LoadKUMulti(args.data_type, args.path_to_data)
    MOVE_TYPES = ("multigrasp", "reaching", "twist")
    ACTION_TYPES = ("realMove", "MI")
    sub_ID_list = range(args.subject_start, args.subject_stop + 1)
    for subject_ID in sub_ID_list:
        print(f"Subject {subject_ID} preprocessing starting")
        start = time.time()
        prep_subject_data = DataLoadingUtils.PrepareData.GetSubjectDataSequential(subject_ID, load_data, args.need_filtering, args.freqlp, args.freqhp, args.path_to_save,
                             args.filter_name, args.need_decimate, args.create_mesh, args.mesh_dim, args.transform_type,
                             args.v73, args.n_cpus_max, args.path_to_vertices)
        prep_subject_data.main_process_ku_multiclass(move_types=MOVE_TYPES, action_types=ACTION_TYPES)
        print(f"Time taken for subject {subject_ID}: {time.time()-start}s")


if __name__ == "__main__":
    main()