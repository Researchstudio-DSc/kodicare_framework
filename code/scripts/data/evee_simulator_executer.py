import hydra

from code.data.evee_simulator import EvEESimulator
from code.utils.io_util import *


@hydra.main(version_base=None, config_path="../../../conf", config_name=None)
def main(cfg):
    docno_date_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.docno_date_path))
    n_evee = cfg.dtc.n_evee
    evee_overlap = cfg.dtc.evee_overlap
    output_path = join(cfg.config.root_dir, join(cfg.dtc.evaluation_splits_dir, cfg.dtc.evee_info_path))

    evee_simulator = EvEESimulator(docno_date_path, n_evee, evee_overlap, output_path)
    evee_simulator.simulate_evee()


if __name__ == '__main__':
    main()
