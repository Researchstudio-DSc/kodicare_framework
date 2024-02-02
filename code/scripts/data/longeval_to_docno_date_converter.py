"""
python script that takes as an input list of files that contain information about: docid, created at, updated at and urls
of the longeval test collection and  creates a csv of ordered files with created at to be used later for the simulation of the EvEE
"""

import hydra
import pandas as pd

from code.utils.io_util import *


def create_info_maps(input_dir, files):
    urls = set()
    info_maps = []
    for file in files:
        lines = read_file_into_list(join(input_dir, file))
        for line in lines:
            info = line.split(sep='\t')
            url = info[1]
            # we consider here only the first occurrence of the document
            if url in urls:
                continue
            info_maps.append({"docno": info[0], "created_at": info[2], "url": url})
    sorted_by_created_at = sorted(info_maps, key=lambda x: x['created_at'])
    return sorted_by_created_at


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    sorted_by_created_at = create_info_maps(cfg.config.root_dir, cfg.dtc.doc_ids__time__infos)
    df = pd.DataFrame(data=sorted_by_created_at)

    df.to_csv(join(cfg.config.root_dir, cfg.dtc.doc_id_date_info_path))


if __name__ == '__main__':
    main()
