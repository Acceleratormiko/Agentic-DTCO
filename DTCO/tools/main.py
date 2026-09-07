from tcad_to_dataset import *
from csv_to_va import dataset_to_va

def main():

    # 0. 调用TCAD工具 进行仿真生成原始数据 (n_iv, p_iv, n_cv, p_cv各一次)

    # 1. 通过 tcad_to_dataset 模块，完成 TCAD 仿真结果，到 NN model 数据集 的建立

    # n 型 I-V 结果的 plt 文件列表
    plt_n_iv_list = [
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p0_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p1_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p2_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p3_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p4_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p5_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p6_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p7_IdVd_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_Vg_0p8_IdVd_des.plt"
    ]
    # p 型 I-V 结果的 plt 文件列表
    plt_p_iv_list = [
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p0_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p1_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p2_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p3_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p4_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p5_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p6_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p7_IdVd_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_Vg_0p8_IdVd_des.plt"
    ]
    # n 型 C-V 结果的 plt 文件列表
    plt_n_cv_list = [
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p0v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p1v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p2v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p3v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p4v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p5v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p6v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p7v_ac_des.plt",
        "3d_n_finfet_HKMG_7nm_NN/result_vg_0p8v_ac_des.plt"
    ]
    
    # p 型 C-V 结果的 plt 文件列表
    plt_p_cv_list = [
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p0v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p1v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p2v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p3v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p4v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p5v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p6v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p7v_ac_des.plt",
        "3d_p_finfet_HKMG_7nm_NN/result_vg_0p8v_ac_des.plt"
    ]

    n_iv_csv, p_iv_csv, n_cv_csv, p_cv_csv = build_nn_dataset(
        plt_n_iv_list,
        plt_p_iv_list,
        plt_n_cv_list,
        plt_p_cv_list,
        filelist = ["n_iv.csv", "p_iv.csv", "n_cv.csv", "p_cv.csv"]
    )

    print("生成的 CSV 绝对路径：")
    print("  n_iv :", n_iv_csv)
    print("  p_iv :", p_iv_csv)
    print("  n_cv :", n_cv_csv)
    print("  p_cv :", p_cv_csv)

    # 2. 将 数据集 输入给 dataset_to_va 模块，内部会完成数据的训练和VA模型的格式转换，最终得到VA模型
    models_dir = "./tmp"
    p_va_path, n_va_path = dataset_to_va(
        n_iv_csv=n_iv_csv,
        p_iv_csv=p_iv_csv,
        n_cv_csv=n_cv_csv,
        p_cv_csv=p_cv_csv,
        models_dir=models_dir,
    )

    print("生成的 VA 模型路径：")
    print("  p_va_path:", p_va_path)
    print("  n_va_path:", n_va_path)

    # 3. VA模型输入给后续的 HSPICE 工具 完成电路评估

if __name__ == "__main__":
    main()


