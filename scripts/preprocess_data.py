"""
Preprocesses data and references from the raw form to a format for modeling
and benchmarking.
Data is read from ../data/{DATASET}
Processed data is written to ../data/processed/{PROCESSED_NAME}
"""

import click
from collections import defaultdict 
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import shutil

@click.command()
@click.option("--ds-name")
@click.option("--dont-remap", is_flag=True, default=False)
def main(ds_name, dont_remap):
    out_path = "../data/processed/" + ds_name
    log_path = out_path + "/preprocess_data.log"
    print(log_path)
    logging.basicConfig(
        filename=log_path,
        encoding="utf-8",
        filemode='w',
        level=logging.DEBUG)
    do_remap = not dont_remap 
    if ds_name in _raw_data_paths:
        preprocess_data(ds_name, enforce_bait_remapping=do_remap)
    elif ds_name in _raw_reference_paths:
        preprocess_reference(ds_name)
    else:
        raise ValueError(f"Unkown dataset name: {ds_name}")

def bait_parser(x, strat):
    if strat == "first_four":
        return x[0:4]
    elif strat == "all":
        return x
    else:
        raise ValueError(f"unkonwn strategy {strat}")


def get_spec_table_and_composites_TODO():
    
colnames = []
prey_names = []

from collections import defaultdict


seen_control_tables = []
keep_controls = []
keep_experiments = []
for sheet_number in range(3):
    df = pd.read_excel("../../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx", sheet_name=sheet_number, header=0)
    for bait in set(df['Bait'].values):
        bait_name = bait.removesuffix("_MG132")
        for i in range(4):
            colnames.append(bait_name + f"_r{i}")
        for i in range(12):
            colnames.append(bait_name + f"_c{i}")
        sel = df['Bait'] == bait
        for i, r in df[sel].iterrows():
            prey_name = r['PreyGene'].removesuffix("_HUMAN")
            e_colname = bait_name + f"_r"
            c_colname = bait_name + f"_c"
            evals = [int(k) for k in r['Spec'].split("|")]
            cvals = [int(k) for k in r['ctrlCounts'].split("|")]

            # What controls to keep?
            experiment_columns = [e_colname + str(i) for i in range(4)]
            outdf.loc[prey_name, experiment_columns] = evals
            control_columns = [c_colname + str(i) for i in range(12)]
            outdf.loc[prey_name, control_columns] = cvals
        control_table = df.loc[sel, ["PreyGene", "ctrlCounts"]]
        keep_experiments += experiment_columns
        if len(seen_control_tables) == 0:
            # Keep the controls
            keep_controls += control_columns
            seen_control_tables.append(control_table)
        else:
            for seen_table in seen_control_tables:
                temp = None
                # Check the table against seen tables at the prey intersections
                shared_prey = set(control_table['PreyGene'].values).intersection(
                    set(seen_table['PreyGene'].values))
                shared_prey = list(shared_prey)
                # If the control tables are equal don't keep
                sel1 = [k in shared_prey for k in control_table['PreyGene'].values]
                sel2 = [k in shared_prey for k in seen_table['PreyGene'].values]

                t1 = control_table[sel1].copy()
                t2 = seen_table[sel2].copy()

                # Sort both tables the same way
                t1 = t1.sort_values("PreyGene")
                t2 = t2.sort_values("PreyGene")

                assert np.alltrue(t1['PreyGene'].values == t2['PreyGene'].values)
                if np.alltrue(t1['ctrlCounts'].values == t2['ctrlCounts'].values):
                    # Do not keep the control columns
                    break
                else:
                    temp = control_table
            # if temp is set then add to the seen tables
            if temp is not None:
                seen_control_tables.append(control_table)
                keep_controls += control_columns            

# Check each bait and get 
all_columns_to_keep = keep_experiments + keep_controls
outdf = outdf.loc[:, all_columns_to_keep]

def handle_controls(strat, prey_condition_dict, prey, condition, found_controls, sim_dict, spec):
    if strat == "every_four":
        ctrl = prey_condition_dict[prey][condition]["ctrl"]
        if ctrl in found_controls[prey]:
            sim_dict[prey] += (spec) 
        else:
            if (ctrl[0:4] == ctrl[4:8]) and (ctrl[4:8] == ctrl[8:12]):
                ctrl = ctrl[0:4]
            found_controls[prey].append(ctrl)
            sim_dict[prey] += (spec + ctrl)
        return sim_dict, found_controls
    elif strat == "all":
        ctrl = prey_condition_dict[prey][condition]["ctrl"]
        if ctrl in found_controls[prey]:
            sim_dict[prey] += (spec) 
        else:
            found_controls[prey].append(ctrl)
            sim_dict[prey] += (spec + ctrl)
        return sim_dict, found_controls
    else:
        raise ValueError

def prey_parser(prey_name, strat):
    assert isinstance(prey_name, str), prey_name
    if strat == "general":
        prey_name = prey_name.removesuffix("_HUMAN")
        return prey_name
    else:
        raise ValueError

def get_mapping_dict(xlsx_path):
    fpath = Path(xlsx_path).parent / "bait_remapping" 
    keyword = fpath.parent.stem 
    to_path = Path("../data/processed") / keyword
    to_path = to_path / "bait_remapping"
    to_path = str(to_path)
    fpath = str(fpath)
    df = pd.read_csv(fpath, names=['b', 'p'])
    d = {r['b']:r['p'] for i, r in df.iterrows()}
    shutil.copy(fpath, to_path)
    return d
        

 
def get_spec_table_and_composites(xlsx_path, sheet_nums,
                                  header=0,
                                  bait_col_name="Bait",
                                  prey_colname="Prey",
                                  ms_score_colname="SaintScore",
                                  ctrl_col='ctrlCounts',
                                  spec_colname="Spec",
                                  spec_count_sep="|",
                                  bait_parser_strat="first_four",
                                  control_handle_strat="all",
                                  prey_parser_strat="general",
                                  enforce_bait_remapping=True,
                                  return_many = False
                                  ):
    # Make the composite table
    logging.info(f"PARAMS")
    logging.info(f"    In path: {xlsx_path}")
    logging.info(f"    sheet_nums {sheet_nums}")
    logging.info(f"    bait_col_name {bait_col_name}")
    logging.info(f"    prey_colname: {prey_colname}")
    logging.info(f"    ms_score_colname : {ms_score_colname}")
    logging.info(f"    ctrl_col: {ctrl_col}")
    logging.info(f"    spec_colname: {spec_colname}")
    logging.info(f"    spec_count_sep: {spec_count_sep}")
    logging.info(f"    bait_parser_strat: {bait_parser_strat}")
    logging.info(f"    control_handle_strat: {control_handle_strat}")
    logging.info(f"    prey_parser_strat: {prey_parser_strat}")
    logging.info(f"    enforce_bait_rempapping: {enforce_bait_remapping}")
    if enforce_bait_remapping:
        bait_mapping_dict = get_mapping_dict(xlsx_path)
    sid = []
    cb = []
    cp = []
    mscore = []
    sid_dict = {}
    k = 0
    prey_condition_dict = {}
    for sheet_num in range(sheet_nums): 
        logging.info(f"sheet {sheet_num}")
        df = pd.read_excel(xlsx_path, sheet_name=sheet_num, header=header) 
        logging.info(f"DF SHAPE : {df.shape}")
        # Sometimes excel corrupts identifers to dates - we remove these without correction
        sel = []
        for i, r in df.iterrows():
            val = isinstance(r[bait_col_name], str) and isinstance(r[prey_colname], str)
            sel.append(val)
        sel = np.array(sel)
        n_corrupt = np.sum(~sel)
        if n_corrupt > 0:
            logging.warning(f"N corrupt lines : {n_corrupt}")
            for i, r in df.loc[~sel, :].iterrows():
                logging.warning(f"corrupt line {i} : {r[bait_col_name], r[prey_colname], r[ms_score_colname]}")
        df = df[sel] 
        logging.info(f"DF SHAPE : {df.shape}")
        assert isinstance(df, pd.DataFrame)
        for i, r in df.iterrows():
            bait_name = r[bait_col_name]
            assert isinstance(bait_name, str)
            if bait_name not in sid_dict:
                sid_dict[bait_name] = k 
                k += 1
            SID = sid_dict[bait_name]
            bait = bait_parser(bait_name, bait_parser_strat)  
            if enforce_bait_remapping and bait in bait_mapping_dict:
                bait = bait_mapping_dict[bait]
            assert isinstance(bait, str), bait
            sid.append(SID)
            cb.append(bait)
            prey_name = r[prey_colname]
            assert isinstance(prey_name, str), (i, prey_name, r)
            prey = prey_parser(prey_name, prey_parser_strat) 
            cp.append(prey)
            saint = r[ms_score_colname]
            assert 0 <= saint <= 1
            mscore.append(saint)
            if prey not in prey_condition_dict:
                prey_condition_dict[prey] = {} 
            assert SID not in prey_condition_dict[prey]
            spec = r[spec_colname]
            spec = str(spec)
            if spec_count_sep in spec:
                spec = spec.split(spec_count_sep)
            else:
                spec = [spec]
            if ctrl_col:
                ctrl = r[ctrl_col]
                ctrl = str(ctrl)
                if spec_count_sep in ctrl:
                    ctrl = ctrl.split(spec_count_sep)
                else:
                    ctrl = [ctrl]
                prey_condition_dict[prey][SID] = {"spec" : spec, "ctrl" : ctrl}
            else:
                prey_condition_dict[prey][SID] = {"spec" : spec} 
    composites = pd.DataFrame(
                {"SID" : sid, "Bait" : cb,
                 "Prey" : cp, "MSscore" : mscore})
    # Gather the conditions
    found_controls = defaultdict(list) 
    sim_dict = defaultdict(list) 
    for prey in prey_condition_dict:
        for condition in prey_condition_dict[prey]:
            spec = prey_condition_dict[prey][condition]["spec"]
            if ctrl_col:
                sim_dict, found_dict = handle_controls(
                        control_handle_strat,
                        prey_condition_dict,
                        prey,
                        condition,
                        found_controls,
                        sim_dict,
                        spec) 
            else:
                sim_dict[prey] += (spec)
    max_conditions = 0
    for prey, conditions in sim_dict.items():
        max_conditions = max(max_conditions, len(conditions))
    nprey = len(sim_dict)
    spec_table = np.zeros((nprey, max_conditions), dtype=int)
    index = []
    for z, (prey, conditions) in enumerate(sim_dict.items()):
        assert len(conditions) > 0
        index.append(prey)
        conditions = np.array([int(i) for i in conditions])
        assert np.alltrue(conditions >= 0)
        spec_table[z, 0:len(conditions)] = conditions 
    spec_table = pd.DataFrame(spec_table, index = index) 
    if return_many:
        return spec_table, composites, prey_condition_dict, sid_dict, sim_dict
    else:
        return spec_table, composites

def log_unmapped_bait(d):
    bait = set(d['Bait'])
    prey = set(d['Prey'])
    u = bait - prey
    logging.info(f"N Bait {len(bait)}")
    if len(u) > 0:
        logging.warning(f"{len(u)} bait not found in prey")
        logging.warning(f"unfound bait {u}")

def preprocess_cullin(prey_colname="PreyGene", enforce_bait_remapping=False):
    spec_table, composites = get_spec_table_and_composites(
            _raw_data_paths["cullin"],
            3,
            prey_colname=prey_colname,
            enforce_bait_remapping=enforce_bait_remapping) 
    log_unmapped_bait(composites)
    composites.to_csv("../data/processed/cullin/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/cullin/spec_table.tsv", sep="\t", index=True, header=False)

def preprocess_dub(prey_colname="Prey", enforce_bait_remapping=False):
    spec_table, composites = get_spec_table_and_composites(
            _raw_data_paths["dub"], 1,
            ctrl_col=None,
            header=2,
            ms_score_colname="SAINT",
            bait_parser_strat="all",
            enforce_bait_remapping=enforce_bait_remapping)
    log_unmapped_bait(composites)
    composites.to_csv("../data/processed/dub/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/dub/spec_table.tsv", sep="\t", index=True, header=False)

def preprocess_tip49(prey_colname="Prey", enforce_bait_remapping=False):
    spec_table, composites = get_spec_table_and_composites(
            _raw_data_paths["tip49"], sheet_nums=1, 
            header=2, ms_score_colname="SAINT",
            bait_parser_strat="all",
            prey_colname=prey_colname,
            enforce_bait_remapping=enforce_bait_remapping)
    log_unmapped_bait(composites)
    composites.to_csv("../data/processed/tip49/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/tip49/spec_table.tsv", sep="\t", index=True, header=False)

def preprocess_sars2(prey_colname="PreyGene", enforce_bait_remapping=False):
    spec_table, composites = get_spec_table_and_composites(
            _raw_data_paths["gordon_sars_cov_2"], 1,
            prey_colname=prey_colname,
            ctrl_col="CtrlCounts",
            header=0,
            ms_score_colname="SaintScore",
            bait_parser_strat="all",
            enforce_bait_remapping=enforce_bait_remapping)
    log_unmapped_bait(composites)
    composites.to_csv("../data/processed/gordon_sars_cov_2/composite_table.tsv", sep="\t", index=False)
    spec_table.to_csv("../data/processed/gordon_sars_cov_2/spec_table.tsv", sep="\t", index=True, header=False)


#References 

def preprocess_biogrid():
    ...

def preprocess_huri():
    ...

def preprocess_humap2():
    ...

def preprocess_data(ds_name, enforce_bait_remapping):
    _preprocess_data[ds_name](enforce_bait_remapping=enforce_bait_remapping)

def preprocess_reference(ds_name):
    _preprocess_reference[ds_name]()


_preprocess_data = {
    "dub" : preprocess_dub,
    "cullin" : preprocess_cullin,
    "tip49" : preprocess_tip49,
    "gordon_sars_cov_2" : preprocess_sars2,
    }

_preprocess_reference = {
    "biogrid" : preprocess_biogrid,
    "huri" : preprocess_huri,
    "huMAP2": preprocess_humap2
        }

_raw_data_paths = {
    "dub" : "../data/dub/41592_2011_BFnmeth1541_MOESM593_ESM.xls",
    "cullin" : "../data/cullin/1-s2.0-S1931312819302537-mmc2.xlsx",
    "tip49" : "../data/tip49/NIHMS252278-supplement-2.xls",
    "gordon_sars_cov_2" : "../data/gordon_sars_cov_2/41586_2020_2286_MOESM5_ESM.xlsx",
        }

_raw_reference_paths = {
        "biogrid" : "../data/biogrid",
        "huri" : "../data/huri",
        "huMAP2" : "../data/humap2",
}

if __name__ == "__main__":
    main()
