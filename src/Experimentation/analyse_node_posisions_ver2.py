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
# from matplotlib import rc
# plt.tight_layout()
# rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
# rc('text', usetex=True)
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

def load_nodes_dispersion(file_name='nodes_dispersion.pikle'):
    nodes_dispersion = joblib.load(os.path.join(dump_detected_testcase_dir, file_name))
    nodes_dispersion = np.array(list(i if i is not None else np.nan for i in nodes_dispersion), dtype=np.float32)
    return nodes_dispersion

def calculate_nodes_dispersion(file_name='nodes_positions.pikle'):
    nodes_position = joblib.load(os.path.join(dump_detected_testcase_dir, file_name))
    import make_database.mesh_interpolation as m_in
    nodes_dispersion = list(m_in.get_nodes_dispersion(*vs, imgQ=np.zeros((600, 800), np.uint8)) for vs in nodes_position)
    nodes_dispersion = np.array(list(i if i is not None else np.nan for i in nodes_dispersion), dtype=np.float32)
    return nodes_dispersion

def mold2Longitude_Latitude(filename):
    filename = filename.replace('_', '-')
    tmp = filename.split('-')
    LoLa = '{0}, {1}'.format(tmp[0], tmp[1])
    return LoLa

if __name__ == "__main__":
    expt_path = os.path.join(myfsys.get_dir_full_path_('outputs'), "expt_split_affinesim_combinable_interporation")
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


    dict_columns = {
        'od': 'our_method-dispersion',
        'ad': 'adaptive_method-dispersion',
        'nd': 'nomethod-dispersion',
        'on': 'our_method-number',
        'an': 'adaptive_method-number',
        'nn': 'nomethod-number',
        'case': 'testcase',
        'set': 'testset'
    }
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
        orig_means = []
        orig_num = []
        nm_means = []
        nm_num = []
        testcase_ticks = []
        for testcase_fn in testcases_fn:
            fn, ext = os.path.splitext(testcase_fn)
            dump_detected_testcase_dir = myfsys.setup_output_directory(dump_detected_dir, fn)
            # nodes_dispersion = calculate_nodes_dispersion()
            # orig_nodes_dispersion = calculate_nodes_dispersion('orig_nodes_positions.pikle')
            # nm_nodes_dispersion = calculate_nodes_dispersion('nonmethod_nodes_positions.pikle')
            nodes_dispersion = load_nodes_dispersion('nodes_dispersion.pikle')
            orig_nodes_dispersion = load_nodes_dispersion('orig_nodes_dispersion.pikle')
            nm_nodes_dispersion = load_nodes_dispersion('nonmethod_nodes_dispersion.pikle')
            Hs = joblib.load(os.path.join(dump_detected_testcase_dir, 'estimated_Hs.pikle'))
            orig_Hs = joblib.load(os.path.join(dump_detected_testcase_dir, 'original_good_Hs.pikle'))
            nm_Hs = joblib.load(os.path.join(dump_detected_testcase_dir, 'nonmethod_good_Hs.pikle'))
            mesh_num = list(True if h is not None else False for h in Hs)
            orig_mesh_num = list(True if h is not None else False for h in orig_Hs)
            nm_mesh_num = list(True if h is not None else False for h in nm_Hs)
            means.append(np.nanmean(nodes_dispersion))
            orig_means.append(np.nanmean(orig_nodes_dispersion))
            nm_means.append(np.nanmean(nm_nodes_dispersion))
            num.append(sum(mesh_num))
            orig_num.append(sum(orig_mesh_num))
            nm_num.append(sum(nm_mesh_num))
            testcase_ticks.append(mold2Longitude_Latitude(fn))
        df = pd.DataFrame()
        df[dict_columns['od']] = means
        df[dict_columns['ad']] = orig_means
        df[dict_columns['nd']] = nm_means
        df[dict_columns['on']] = num
        df[dict_columns['an']] = orig_num
        df[dict_columns['nn']] = nm_num
        df[dict_columns['case']] = testcase_ticks
        df[dict_columns['set']] = testset_name
        df_means = pd.DataFrame({
            testset_name: means
        })
        df_orig_means = pd.DataFrame({
            testset_name: orig_means
        })
        df_nm_means = pd.DataFrame({
            testset_name: nm_means
        })
        df_nums = pd.DataFrame({
            testset_name: num
        })
        df_orig_nums = pd.DataFrame({
            testset_name: orig_num
        })
        df_nm_nums = pd.DataFrame({
            testset_name: nm_num
        })
        dfs = pd.concat([dfs, df])
        dfs_means = pd.concat([dfs_means, df_means], sort=False)
        dfs_means = pd.concat([dfs_means, df_orig_means], sort=False)
        dfs_means = pd.concat([dfs_means, df_nm_means], sort=False)
        dfs_nums = pd.concat([dfs_nums, df_nums], sort=False)
        dfs_nums = pd.concat([dfs_nums, df_orig_nums], sort=False)
        dfs_nums = pd.concat([dfs_nums, df_nm_nums], sort=False)
        # ax1 = df.plot(kind='bar', y=['our_method-number', 'nomethod-number'],
        #               xticks=df.index, grid=True, rot=30, title="All Detected Mesh Number - {}".format(testset_name))
        # # ax1.set_xlim((0, len(df.index)))
        # _ylim=(0, 64)
        # ax1.set_ylim(_ylim)
        # # ax1.set_yticks(_yticks)
        # # _ylim=(df['position_means'].min(), df['position_means'].max())
        # # _ylim=(0, df['position_means'].max())
        # # _yticks=np.linspace(df['position_means'].min(), df['position_means'].max(), 10, endpoint=True)
        # # _yticks=np.linspace(0, 1.0, 10, endpoint=True)
        # ax1.set_xticklabels(df.testcase)
        # ax1.set_ylabel("Mesh number")
        # ax1.set_xlabel("Longitude, Latitude")
        # # ax1.set_ylim([df['position_means'].min, df['position_means'].max])
        # # _ylim2=[df['mesh_nums'].min(), df['mesh_nums'].max()]
        # # _ylim2=(0, 64)
        # ax1.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
        # # ax1.legend(loc='uppper right',
        # #            bbox_to_anchor=(1.05, 0.5, 0.5, .100),
        # #            borderaxespad=0.,
        # #            ncol=1,
        # #            mode="expand",
        # #            title="LABEL NAME")
        # fig_1 = ax1.get_figure()
        # fig_1.tight_layout()
        # fig_1.savefig(os.path.join(plot_graph, 'numbers_' + testset_name + ".png"))
        # sleep(5)
        #
        # ax2 = df.plot(kind='bar', y=[dict_columns['od'], dict_columns['nd']],
        #               xticks=df.index, grid=True, rot=30, title="Vertexes Dispersion Means - {}".format(testset_name))
        # # ax2.set_ylim(_ylim2)
        # # ax2.set_xlim((0, len(df.index)-1))
        # _yticks=np.arange(0, 11)/10
        # ax2.set_yticks(_yticks)
        # ax2.set_xticklabels(df.testcase)
        # ax2.set_ylabel("Vertex dispersion")
        # ax2.set_xlabel("Longitude, Latitude")
        # # ax2.tight_layout()
        # # ax2.set_ylim([df['mesh_nums'].min, df['mesh_nums'].max])
        # # ax2.legend(loc='uppper right',
        # #            bbox_to_anchor=(1.05, 0.5, 0.5, .100),
        # #            borderaxespad=0.,
        # #            ncol=1,
        # #            mode="expand",
        # #            title="LABEL NAME")
        # #, fontsize=20, xticks=testcase_ticks, title=testset_name
        # fig = ax2.get_figure()
        # fig.tight_layout()
        # fig.savefig(os.path.join(plot_graph, 'means_' + testset_name + ".png"))
        # sleep(5)
    output_dir_plot = myfsys.setup_output_directory(expt_name, 'polot_graph')
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


    # set = set(dfs.testset)
    # import itertools
    # for combi in itertools.combinations(set, 2):
    #     d_ylim = (0, 1)
    #     d_yticks = np.arange(0, 11)/10
    #     dispersion_df_combi = dfs_means[dfs_means.testset == combi[0] or dfs_means.testset == combi[1]][
    #         [dict_columns['om'], dict_columns['nm']]]
    #     dispersion = dfs_means.plot(kind='bar', grid=True,
    #                          rot=30, title="Vertexes Dispersion Means - {} & {}")
    #     # ax3.set_xlim((0, len(df.index)-1))
    #     ax3.set_xticks(df.index)
    #     ax3.set_xticklabels(df.testcase)
    #     ax3.set_ylim(_ylim)
    #     ax3.set_yticks(_yticks)
    #     ax3.set_ylabel("Vertex position dispersion")
    #     ax3.set_xlabel("camera position No._Lon.-Lat.")
    #     fig3 = ax3.get_figure()
    #     fig3.tight_layout()
    #     fig3.savefig(os.path.join(output_dir_plot, "Means.png"))
    #     sleep(5)
    #
    #     _ylim2=(0, 64)
    #     ax4 = dfs_nums.plot(kind='bar', grid=True,
    #                         rot=30, title="Estimated Meshe Number - All Shape")
    #     ax4.set_ylim(_ylim2)
    #     ax4.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
    #     # ax4.set_xlim((0, len(df.index)-1))
    #     ax4.set_xticks(df.index)
    #     ax4.set_xticklabels(df.testcase)
    #     ax4.set_ylabel("Estimated mesh number")
    #     ax4.set_xlabel("camera position No._Lon.-Lat.")
    #     fig4 = ax4.get_figure()
    #     fig4.tight_layout()
    #     fig4.savefig(os.path.join(output_dir_plot, "Numbers.png"))
    #     sleep(5)



    _ylim=(0, 1)
    _yticks=np.arange(0, 11)/10
    df3 = dfs_means[dfs_means.testset == 'hoge'][[dict_columns['om'], dict_columns['nm']]]
    ax3 = dfs_means.plot(kind='bar', xticks=df.index, grid=True,
                        rot=30, title="Estimated Vertexes Position Means - All Shape")
    # ax3.set_xlim((0, len(df.index)-1))
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
    ax4 = dfs_nums.plot(kind='bar', grid=True,
                        rot=30, title="Estimated Meshe Number - All Shape")
    ax4.set_ylim(_ylim2)
    ax4.set_yticks([0, 10, 20, 30, 40, 50, 60, 64])
    # ax4.set_xlim((0, len(df.index)-1))
    ax4.set_xticks(df.index)
    ax4.set_xticklabels(df.testcase)
    ax4.set_ylabel("Estimated mesh number")
    ax4.set_xlabel("camera position No._Lon.-Lat.")
    fig4 = ax4.get_figure()
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir_plot, "Numbers.png"))
    sleep(5)


