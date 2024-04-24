# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spec_table_path = "../se_sr_all_20k/spec_table.tsv"
pread = lambda x: pd.read_csv(x, sep="\t", index_col=0)
st = pread(spec_table_path)
# -

rsel = ['CUL5','CUL2', 'PEBB', 'DCA11', 'vifprotein', 'ELOB', 'ELOC', 'LLR1', 'CSN2', 'CSN5','MYH14',
       'ANM5']
rsel2 = rsel.copy()
rsel2.remove('vifprotein')


def spec_table_pretty_plot(st, name, rot=90):
    xnames = {"ELOBwt_r0": "ELOB WT", "CUL5wt_r0" : "CUL5 WT",
              "CBFBwt_r0": "CBFB WT", "ELOBvif_r0" : "ELOB Vif"}
    
    labels = [k if i % 4 == 0 else "" for i, k in enumerate(st.columns)]
    yticklabels = [k if k != "PEBB" else "CBFB" for k in rsel]
    display_frame = st.copy()
    def suffixer(name):
        if 'wt'in name:
            name = name.removesuffix('wt') + " WT"
        elif 'vif' in name:
            name = name.removesuffix('vif') + ' Vif'
        elif 'mock'in name:
            name = name.removesuffix('mock') + ' Mock'
        return name
    
    display_frame_columns = ['C' if '_c' in name else name for name in display_frame.columns]
    display_frame_columns = [i.split("_")[0] for i in display_frame_columns]
    display_frame_columns = [suffixer(name) for name in display_frame_columns]

    
    display_frame.columns = display_frame_columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(display_frame, cmap='coolwarm', xticklabels=4, yticklabels=yticklabels,
               cbar_kws={'label': 'Spectral counts'})
    plt.xticks(rotation=rot)
    plt.gcf().subplots_adjust(bottom=0.15)
    
    plt.savefig(name + "_300.png", dpi=300)
    plt.savefig(name + "_1200.png", dpi=1200)

    plt.show()
    plt.close()

spec_table_pretty_plot(st.loc[rsel, :], name="se_sr_all_20k")

st_wt = pread("../se_sr_wt_ctrl_20k/spec_table.tsv")
spec_table_pretty_plot(st_wt.loc[rsel, :], name="se_sr_wt_20k")

st_vif = pread("../se_sr_vif_ctrl_20k/spec_table.tsv")
spec_table_pretty_plot(st_vif.loc[rsel, :], name = "se_sr_vif_20k")

# +

st_mock = pread("../se_sr_mock_ctrl_20k/spec_table.tsv")
spec_table_pretty_plot(st_mock.loc[rsel2, :], name="se_sr_mock_20k")
