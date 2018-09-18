import os
from commons import my_file_path_manager as myfsys
from commons import expt_modules as emod
from commons.expt_modules import PrefixShapes as prfx
from commons.custom_find_obj import init_feature
import itertools
import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.tight_layout()
import pandas as pd
from time import sleep
import cv2

def get_testset(*args):
    return args[0] + args[1]

def get_testset(keyword_set, template_fn=None):
    if template_fn is None:
        for prefix, tmp_fn in keyword_set:
            yield {'prefix_shape': prefix, 'template_fn': tmp_fn}
    else:
        for prefix, tmp_fn in keyword_set:
            if template_fn == tmp_fn:
                yield {'prefix_shape': prefix, 'template_fn': tmp_fn}

def get_testcase_fn(testset_fullpath, sampling_num=0):
    testcase_fns = os.listdir(testset_full_path)
    testcase_fns.sort()
    if not sampling_num == 0:
        for i, test_case_fn in enumerate(testcase_fns):
            if i % sampling_num == 0:
                yield test_case_fn
    else:
        for i, test_case_fn in enumerate(testcase_fns):
            yield test_case_fn

if __name__ == "__main__":
    expt_path = os.path.join(myfsys.get_dir_full_path_('outputs'), "expt_mesh_interpolation")
    detector, matcher = init_feature(emod.Features.SIFT.name)
    column_num = 8
    row_num = 8
    split_num = column_num * row_num
    expt_name = os.path.basename(expt_path)

    prefixes = (prfx.PL.value,
                prfx.MLTF.value,
                prfx.CRV.value)
    tmp_fn_list = ('qrmarker.png',
                   'nabe.png')
    #tmp_fn_list = emod.only(tmp_fn_list, 'glass.png')
    tmp_name = tuple(os.path.splitext(fn)[0] for fn in tmp_fn_list)
    testset_set = list(itertools.product(prefixes, tmp_fn_list)) #2set product
    sampling_num = 57
    gene = get_testset(testset_set)
    dfs = pd.DataFrame()
    dfs_means = pd.DataFrame()
    dfs_nums = pd.DataFrame()
    for testset in gene:
        testset_full_path = myfsys.get_dir_full_path_testset('cgs', **testset)
        testset_name = os.path.basename(testset_full_path)
        output_dir = myfsys.setup_output_directory(expt_name, testset_name)
        input_dir = myfsys.setup_output_directory(output_dir, 'dump_detected_dir')
        plot_graph = myfsys.setup_output_directory(output_dir, 'plot_graph')
        dump_detected_dir = myfsys.setup_output_directory(output_dir, 'dump_detected_dir')
        testcases_fn = get_testcase_fn(testset_full_path, sampling_num=sampling_num)
        means = []
        num = []
        testcase_ticks = []
        for testcase_fn in testcases_fn:
            fn, ext = os.path.splitext(testcase_fn)
            dump_detected_testcase_dir = myfsys.setup_output_directory(dump_detected_dir, fn)
            nodes_dispersion = joblib.load(os.path.join(dump_detected_testcase_dir, 'nodes_dispersion.pikle'))
            nodes_dispersion = np.array(list(i if i is not None else np.nan for i in nodes_dispersion), dtype=np.float32)
            good_Hs = joblib.load(os.path.join(dump_detected_testcase_dir, 'good_Hs.pikle'))
            mesh_num = list(True if h is not None else False for h in good_Hs)
            means.append(np.nanmean(nodes_dispersion))
            num.append(sum(mesh_num))
            testcase_ticks.append(fn)

        df = pd.DataFrame({
            'position_means': means,
            'mesh_nums': num,
            'testcase': testcase_ticks,
            'testset': testset_name
        })
        df_means = pd.DataFrame({
            testset_name: means
        })
        df_nums = pd.DataFrame({
            testset_name: num
        })
        dfs = pd.concat([dfs, df])
        dfs_means = pd.concat([dfs_means, df_means], sort=False)
        dfs_nums = pd.concat([dfs_nums, df_nums], sort=False)
        # # _ylim=(df['position_means'].min(), df['position_means'].max())
        # # _ylim=(0, df['position_means'].max())
        # _ylim=(0, 1)
        # # _yticks=np.linspace(df['position_means'].min(), df['position_means'].max(), 10, endpoint=True)
        # _yticks=np.linspace(0, 1.0, 10, endpoint=True)
        # ax1 = df.plot(kind='line', y=df.columns[1], xticks=df.index)
        # ax1.set_xlim((0, len(df.index)))
        # ax1.set_ylim(_ylim)
        # ax1.set_yticks(_yticks)
        # ax1.set_ylabel("Vertex position dispersion")
        # ax1.set_xlabel("camera position No._Lon.-Lat.")
        # # ax1.set_ylim([df['position_means'].min, df['position_means'].max])
        # # _ylim2=[df['mesh_nums'].min(), df['mesh_nums'].max()]
        # _ylim2=(0, 64)
        # ax2 = df.plot(kind='line', y=df.columns[0], grid=True,
        #               ax=ax1, secondary_y=True, rot=30, title=testset_name)
        # ax2.set_ylim(_ylim2)
        # ax2.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
        # ax2.set_xlim((0, len(df.index)-1))
        # ax2.set_xticks(df.index)
        # ax2.set_xticklabels(df.testcase)
        # ax2.set_ylabel("Estimated mesh number")
        # # ax2.tight_layout()
        # # ax2.set_ylim([df['mesh_nums'].min, df['mesh_nums'].max])
        # #, fontsize=20, xticks=testcase_ticks, title=testset_name
        # fig = ax2.get_figure()
        # fig.tight_layout()
        # fig.savefig(os.path.join(plot_graph, testset_name + ".png"))
        # sleep(5)

    output_dir_plot= myfsys.setup_output_directory(expt_name, 'polot_graph')
    gene = get_testset(testset_set)
    shapes = ['Plane',
              'Multifaceted',
              'Curved']
    # for i, pref in enumerate(prefixes):
    #     _ylim=(0, 1)
    #     _yticks=np.linspace(0, 1.0, 10, endpoint=True)
    #     ax1 = dfs_means.plot(kind='line', y=[pref+'qrmarker', pref+'nabe'], xticks=df.index)
    #     ax1.set_xlim((0, len(df.index)))
    #     ax1.set_ylim(_ylim)
    #     ax1.set_yticks(_yticks)
    #     ax1.set_ylabel("Vertex position dispersion")
    #     ax1.set_xlabel("camera position No._Lon.-Lat.")
    #
    #     _ylim2=(0, 64)
    #     ax2 = dfs_nums.plot(kind='line', y=[pref+'qrmarker', pref+'nabe'], grid=True,
    #                   ax=ax1, secondary_y=True, rot=30, title=shapes[i])
    #     ax2.set_ylim(_ylim2)
    #     ax2.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
    #     ax2.set_xlim((0, len(df.index)-1))
    #     ax2.set_xticks(df.index)
    #     ax2.set_xticklabels(df.testcase)
    #     ax2.set_ylabel("Estimated mesh number")
    #     fig = ax2.get_figure()
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(output_dir_plot, shapes[i] + ".png"))
    #     sleep(5)

    _ylim=(0, 1)
    _yticks=np.linspace(0, 1.0, 10, endpoint=True)
    ax3 = dfs_means.plot(kind='line', xticks=df.index, grid=True,
                        rot=30, title="Estimated Vertexes Position Means - All Shape")
    ax3.set_xlim((0, len(df.index)-1))
    ax3.set_xticks(df.index)
    ax3.set_xticklabels(df.testcase)
    ax3.set_ylim(_ylim)
    ax3.set_yticks(_yticks)
    ax3.set_ylabel("Vertex position dispersion")
    ax3.set_xlabel("camera position No._Lon.-Lat.")
    fig3 = ax3.get_figure()
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir_plot, "Means.png"))
    sleep(5)

    _ylim2=(0, 64)
    ax4 = dfs_nums.plot(kind='line', grid=True,
                        rot=30, title="Estimated Meshes Number - All Shape")
    ax4.set_ylim(_ylim2)
    ax4.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
    ax4.set_xlim((0, len(df.index)-1))
    ax4.set_xticks(df.index)
    ax4.set_xticklabels(df.testcase)
    ax4.set_ylabel("Estimated mesh number")
    ax4.set_xlabel("camera position No._Lon.-Lat.")
    fig4 = ax4.get_figure()
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir_plot, "Numbers.png"))
    sleep(5)


