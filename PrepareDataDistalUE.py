import os
import time
import argparse
import numpy as np
import DataLoadingUtils.PrepareData
import DataLoadingUtils.LoadDistalUE

def main():
    parser = argparse.ArgumentParser()
    parser = DataLoadingUtils.PrepareData.get_args(parser)
    parser.add_argument("--data_type", default="vhdr", help="This is currently the only available format of RAW data")
    parser.add_argument("--version", default=2, type=int, help="1: trial-based data, 2: continuous movement")
    args = parser.parse_args()
    data_loader = DataLoadingUtils.LoadDistalUE.LoadDistalUE(args.version, args.data_type, args.path_to_data)
    MOVE_TYPES = ("distalUE")
    ACTION_TYPES = ["realMove"] if args.version == 2 else ("realMove", "MI")
    sub_ID_list = range(args.subject_start, args.subject_stop + 1)
    for subject_ID in sub_ID_list:
        print(f"Subject {subject_ID}, preprocessing starting")
        start = time.time()
        prep_subject_data = DataLoadingUtils.PrepareData.GetSubjectDataSequential(subject_ID, data_loader,
                                                                                  args.need_filtering, args.freqlp,
                                                                                  args.freqhp, args.path_to_save,
                                                                                  args.filter_name, args.need_decimate,
                                                                                  args.create_mesh, args.mesh_dim, 1,
                                                                                  args.transform_type,
                                                                                  args.v73, args.n_cpus_max,
                                                                                  args.path_to_vertices)
        prep_subject_data.main_process_distalue_multiclass(action_types=ACTION_TYPES)
        print(f"Time taken for subject {subject_ID}: {time.time() - start}s")

if __name__ == "__main__":
    main()