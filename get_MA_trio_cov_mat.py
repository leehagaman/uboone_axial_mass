
import numpy as np
import uproot as uproot
import uproot3 as uproot3
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import pickle

def get_prediction_cv_and_variations_dataframes():

    bdt_vars = [
        "nue_score",
        "numu_score",
        "numu_cc_flag"
    ]

    eval_vars = [
        "run",
        "subrun",
        "event",
        "truth_nuEnergy",
        "truth_nuPdg",
        "truth_isCC",
        "truth_vtxInside",
        "match_isFC",
        "match_completeness_energy",
        "truth_energyInside",

        "weight_cv",
        "weight_spline",
    ]

    eval_data_vars = [
        "match_isFC",
    ]

    kine_vars = [
        "kine_reco_Enu",
    ]

    pf_vars = [
        "reco_muonMomentum",
        "truth_muonMomentum",
    ]

    pf_data_vars = [
        "reco_muonMomentum",
    ]

    weight_vars = [
        # the framework never uses these, it uses the ones in T_eval instead!
        #"weight_cv",
        #"weight_spline",

        "All_UBGenie",
        
        "AxFFCCQEshape_UBGenie",
        "DecayAngMEC_UBGenie",
        "NormCCCOH_UBGenie",
        "NormNCCOH_UBGenie",
        "RPA_CCQE_UBGenie",
        "ThetaDelta2NRad_UBGenie",
        "Theta_Delta2Npi_UBGenie",
        "VecFFCCQEshape_UBGenie",
        "XSecShape_CCMEC_UBGenie",
        "xsr_scc_Fa3_SCC",
        "xsr_scc_Fv3_SCC",
    ]

    #loc = "/Users/leehagaman/data/processed_checkout_rootfiles/"
    loc = "/Users/leehagaman/data/from_london/"

    print("Loading nu overlay run 1 CV")

    f = uproot3.open(loc + "checkout_prodgenie_bnb_nu_overlay_run1.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    nu_overlay_run1_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    nu_overlay_run1_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    nu_overlay_run1_df["file"] = "nu_overlay_run1"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    print("Loading nu overlay run 2 CV")

    f = uproot3.open(loc + "checkout_prodgenie_bnb_nu_overlay_run2.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    nu_overlay_run2_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    nu_overlay_run2_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    nu_overlay_run2_df["file"] = "nu_overlay_run2"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    print("Loading nu overlay run 3 CV")

    f = uproot3.open(loc + "checkout_prodgenie_bnb_nu_overlay_run3.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    nu_overlay_run3_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    nu_overlay_run3_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    nu_overlay_run3_df["file"] = "nu_overlay_run3"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    nu_overlay_df = pd.concat([
        nu_overlay_run1_df, 
        nu_overlay_run2_df, 
        nu_overlay_run3_df], sort=False)

    print("Loading cv dirt overlay run 1")

    f = uproot3.open(loc + "checkout_prodgenie_dirt_overlay_run1_all.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    dirt_run1_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    dirt_run1_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    dirt_run1_df["file"] = "dirt_run1"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    print("Loading cv dirt overlay run 2")

    f = uproot3.open(loc + "checkout_prodgenie_dirt_overlay_run2_all.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    dirt_run2_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    dirt_run2_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    dirt_run2_df["file"] = "dirt_run2"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    print("Loading cv dirt overlay run 3")

    f = uproot3.open(loc + "checkout_prodgenie_dirt_overlay_run3_all.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    dirt_run3_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    dirt_run3_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    dirt_run3_df["file"] = "dirt_run3"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    dirt_df = pd.concat([
        dirt_run1_df, 
        dirt_run2_df, 
        dirt_run3_df], sort=False)

    print("Loading ext run 1")

    f = uproot3.open(loc + "wcp_data_extbnb_run1_mcc9_v08_00_00_53_checkout.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_data_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_data_vars, flatten=False)
    ext_run1_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    ext_run1_df["file"] = "ext_run1"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    print("Loading ext run 2")

    f = uproot3.open(loc + "wcp_data_extbnb_run2_mcc9_v08_00_00_53_checkout.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_data_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_data_vars, flatten=False)
    ext_run2_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    ext_run2_df["file"] = "ext_run2"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    print("Loading ext run 3")

    f = uproot3.open(loc + "wcp_data_extbnb_run3_mcc9_v08_00_00_53_checkout.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_data_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_data_vars, flatten=False)
    ext_run3_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval], axis=1, sort=False)
    ext_run3_df["file"] = "ext_run3"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval

    ext_df = pd.concat([
        ext_run1_df, 
        ext_run2_df, 
        ext_run3_df], sort=False)

    ext_df["weight_cv"] = [1. for _ in range(ext_df.shape[0])]
    ext_df["weight_spline"] = [1. for _ in range(ext_df.shape[0])]
    six_hundred_ones_arr = np.array([1. for _1 in range(600)])
    All_UBGenie_ones_arr = [six_hundred_ones_arr for _2 in range(ext_df.shape[0])]
    ext_df["All_UBGenie"] = All_UBGenie_ones_arr


    print("Loading nu overlay run 1 XS universes")

    f = uproot3.open(loc + "prodgenie_bnb_nu_overlay_run1/UBGenieFluxSmallUni.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    f_weight = f["T_weight"].pandas.df(weight_vars, flatten=False)
    nu_overlay_run1_vars_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    nu_overlay_run1_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval, f_weight], axis=1, sort=False)
    nu_overlay_run1_df["file"] = "nu_overlay_run1"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval
    del f_weight

    print("Loading nu overlay run 2 XS universes")

    f = uproot3.open(loc + "prodgenie_bnb_nu_overlay_run2/UBGenieFluxSmallUni.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    f_weight = f["T_weight"].pandas.df(weight_vars, flatten=False)
    nu_overlay_run2_vars_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    nu_overlay_run2_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval, f_weight], axis=1, sort=False)
    nu_overlay_run2_df["file"] = "nu_overlay_run2"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval
    del f_weight

    print("Loading nu overlay run 3 XS universes")

    f = uproot3.open(loc + "prodgenie_bnb_nu_overlay_run3/UBGenieFluxSmallUni.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    f_weight = f["T_weight"].pandas.df(weight_vars, flatten=False)
    nu_overlay_run3_vars_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    nu_overlay_run3_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval, f_weight], axis=1, sort=False)
    nu_overlay_run3_df["file"] = "nu_overlay_run3"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval
    del f_weight

    nu_overlay_vars_df = pd.concat([
        nu_overlay_run1_df, 
        nu_overlay_run2_df, 
        nu_overlay_run3_df], sort=False)

    print("Setting number for unisim variations")

    num_unisim_variations_dic = {}
    for unisim_type in ["AxFFCCQEshape_UBGenie",
                                        "DecayAngMEC_UBGenie",
                                        "NormCCCOH_UBGenie",
                                        "NormNCCOH_UBGenie",
                                        "RPA_CCQE_UBGenie",
                                        "ThetaDelta2NRad_UBGenie",
                                        "Theta_Delta2Npi_UBGenie",
                                        "VecFFCCQEshape_UBGenie",
                                        "XSecShape_CCMEC_UBGenie",
                                        "xsr_scc_Fa3_SCC",
                                        "xsr_scc_Fv3_SCC"]:

        num_unisim_variations_dic[unisim_type] = len(nu_overlay_vars_df[unisim_type].to_numpy()[0])

    print("adding uniform ext variation info to ext files")

    for unisim_type in ["AxFFCCQEshape_UBGenie",
                                        "DecayAngMEC_UBGenie",
                                        "NormCCCOH_UBGenie",
                                        "NormNCCOH_UBGenie",
                                        "RPA_CCQE_UBGenie",
                                        "ThetaDelta2NRad_UBGenie",
                                        "Theta_Delta2Npi_UBGenie",
                                        "VecFFCCQEshape_UBGenie",
                                        "XSecShape_CCMEC_UBGenie",
                                        "xsr_scc_Fa3_SCC",
                                        "xsr_scc_Fv3_SCC",]:
        
        ext_df[unisim_type] = [np.array([1. for _1 in range(num_unisim_variations_dic[unisim_type])]) for _2 in range(ext_df.shape[0])] 

    print("Loading dirt overlay run 1 XS universes")

    f = uproot3.open(loc + "prodgenie_dirt_overlay_run1_all/UBGenieFluxSmallUni.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    f_weight = f["T_weight"].pandas.df(weight_vars, flatten=False)
    dirt_run1_vars_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    dirt_run1_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval, f_weight], axis=1, sort=False)
    dirt_run1_df["file"] = "dirt_run1"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval
    del f_weight

    print("Loading dirt overlay run 2 XS universes")

    f = uproot3.open(loc + "prodgenie_dirt_overlay_run2_all/UBGenieFluxSmallUni.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    f_weight = f["T_weight"].pandas.df(weight_vars, flatten=False)
    dirt_run2_vars_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    dirt_run2_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval, f_weight], axis=1, sort=False)
    dirt_run2_df["file"] = "dirt_run2"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval
    del f_weight

    print("Loading dirt overlay run 3 XS universes")

    f = uproot3.open(loc + "prodgenie_dirt_overlay_run3_all/UBGenieFluxSmallUni.root")["wcpselection"]
    f_bdt = f["T_BDTvars"].pandas.df(bdt_vars, flatten=False)
    f_eval = f["T_eval"].pandas.df(eval_vars, flatten=False)
    f_kine = f["T_KINEvars"].pandas.df(kine_vars, flatten=False)
    f_pfeval = f["T_PFeval"].pandas.df(pf_vars, flatten=False)
    f_weight = f["T_weight"].pandas.df(weight_vars, flatten=False)
    dirt_run3_vars_pot = np.sum(f["T_pot"].pandas.df("pot_tor875good", flatten=False)["pot_tor875good"].to_numpy())
    dirt_run3_df = pd.concat([f_bdt, f_eval, f_kine, f_pfeval, f_weight], axis=1, sort=False)
    dirt_run3_df["file"] = "dirt_run3"
    del f
    del f_bdt
    del f_eval
    del f_kine
    del f_pfeval
    del f_weight

    dirt_vars_df = pd.concat([
        dirt_run1_df, 
        dirt_run2_df, 
        dirt_run3_df], sort=False)


    print("unifying all dataframes")

    all_df = pd.concat([nu_overlay_df, dirt_df, ext_df], sort=False)
    all_vars_df = pd.concat([nu_overlay_vars_df, dirt_vars_df], sort=False)


    print("calculating reco costheta and muon momentum")

    costheta_vals = []
    muonmomentum_vals = []
    reco_muonmomentum_x = all_df["reco_muonMomentum[0]"].to_numpy()
    reco_muonmomentum_y = all_df["reco_muonMomentum[1]"].to_numpy()
    reco_muonmomentum_z = all_df["reco_muonMomentum[2]"].to_numpy()
    reco_muonmomentum_t = all_df["reco_muonMomentum[3]"].to_numpy()
    for i in range(len(reco_muonmomentum_x)):
        if reco_muonmomentum_t[i] < 105.66 / 1000.: # surprising that this happens for positive values, but I did find some events
            costheta_vals.append(-1)
            muonmomentum_vals.append(-1)
        else:
            costheta_vals.append(reco_muonmomentum_z[i] / np.sqrt(reco_muonmomentum_x[i]**2 + reco_muonmomentum_y[i]**2 + reco_muonmomentum_z[i]**2))
            muon_KE = reco_muonmomentum_t[i] * 1000. - 105.66
            muonmomentum_vals.append(np.sqrt(muon_KE**2 + 2 * muon_KE * 105.66))

    all_df["reco_costheta"] = costheta_vals
    all_df["reco_muon_momentum"] = muonmomentum_vals

    costheta_vals = []
    muonmomentum_vals = []
    reco_muonmomentum_x = all_vars_df["reco_muonMomentum[0]"].to_numpy()
    reco_muonmomentum_y = all_vars_df["reco_muonMomentum[1]"].to_numpy()
    reco_muonmomentum_z = all_vars_df["reco_muonMomentum[2]"].to_numpy()
    reco_muonmomentum_t = all_vars_df["reco_muonMomentum[3]"].to_numpy()
    for i in range(len(reco_muonmomentum_x)):
        if reco_muonmomentum_t[i] < 105.66 / 1000.: # surprising that this happens for positive values, but I did find some events
            costheta_vals.append(-1)
            muonmomentum_vals.append(-1)
        else:
            costheta_vals.append(reco_muonmomentum_z[i] / np.sqrt(reco_muonmomentum_x[i]**2 + reco_muonmomentum_y[i]**2 + reco_muonmomentum_z[i]**2))
            muon_KE = reco_muonmomentum_t[i] * 1000. - 105.66
            muonmomentum_vals.append(np.sqrt(muon_KE**2 + 2 * muon_KE * 105.66))

    all_vars_df["reco_costheta"] = costheta_vals
    all_vars_df["reco_muon_momentum"] = muonmomentum_vals


    print("calculating truth muon momentum and KE")

    all_df["truth_muonMomentum_3"] = all_df["truth_muonMomentum[3]"].to_numpy()
    all_df["true_muon_KE"] = all_df["truth_muonMomentum_3"].to_numpy()*1000.-105.66
    all_df["true_muon_momentum"] = np.sqrt(all_df["true_muon_KE"]**2 + 2*all_df["true_muon_KE"]*105.66)

    all_vars_df["truth_muonMomentum_3"] = all_vars_df["truth_muonMomentum[3]"].to_numpy()
    all_vars_df["true_muon_KE"] = all_vars_df["truth_muonMomentum_3"].to_numpy()*1000.-105.66
    all_vars_df["true_muon_momentum"] = np.sqrt(all_vars_df["true_muon_KE"]**2 + 2*all_vars_df["true_muon_KE"]*105.66)


    pots = [nu_overlay_run1_pot, nu_overlay_run2_pot, nu_overlay_run3_pot, dirt_run1_pot, dirt_run2_pot, dirt_run3_pot, 
            nu_overlay_run1_vars_pot, nu_overlay_run2_vars_pot, nu_overlay_run3_vars_pot, dirt_run1_vars_pot, dirt_run2_vars_pot, dirt_run3_vars_pot]


    return all_df, all_vars_df, num_unisim_variations_dic, pots


##### Function to get the covariance matrix and prediction for a certain type of data or fake data extraction #####


def get_MA_trio_cov_mat_pred(
        
        data_type = "NuWro", # other options are "real" and "GENIE_v2"
        
        #use_real_data = False,
        #use_nuwro_fake_data = False,
        #use_genie_v2_fake_data = False,

        skip_AxFFCCQEshape_UBGenie = False,

        shape_type = "rate+shape", # other options ar "+100" and "matrix_breakdown"

        collapse_type = "4D", # other options are "2D" and "1D"

        no_cache = False,
        ):
    
    assert not data_type == "real", "Not allowed to unblind yet!"

    cache_key = f"cache_{data_type}"
    if skip_AxFFCCQEshape_UBGenie:
        cache_key += f"_skip_AxFFCCQEshape_UBGenie"
    if shape_type != "rate+shape":
        cache_key += f"_{shape_type}"
    if collapse_type != "4D":
        cache_key += f"_{collapse_type}"
    cache_key += ".pkl"

    if not no_cache:
        try:
            print(f"Attempting to load from cache file: {cache_key}")
            with open("trio_caches/" + cache_key, 'rb') as f:
                total_cov_MA, tot_pred_MA, data, multisim_xs_MA_cov, universe_reco_MAs = pickle.load(f)
                print("Successfully loaded from cache")
                return total_cov_MA, tot_pred_MA, data, multisim_xs_MA_cov, universe_reco_MAs
        except (FileNotFoundError, EOFError):
            print("Cache not found or invalid, computing from scratch...")


    all_df, all_vars_df, num_unisim_variations_dic, pots = get_prediction_cv_and_variations_dataframes()

    nu_overlay_run1_pot, nu_overlay_run2_pot, nu_overlay_run3_pot, dirt_run1_pot, dirt_run2_pot, dirt_run3_pot, nu_overlay_run1_vars_pot, nu_overlay_run2_vars_pot, nu_overlay_run3_vars_pot, dirt_run1_vars_pot, dirt_run2_vars_pot, dirt_run3_vars_pot = pots

    print("loading 3D XS extraction files")

    if data_type == "real":
        # commenting to make sure we don't load real data yet
        #f_merged = uproot.open("numuCC_3d_data/real_data/merge_xs_data.root")
        #f_wiener = uproot.open("numuCC_3d_data/real_data/wiener_data.root")
        pass
    elif data_type == "NuWro":
        f_merged = uproot.open("numuCC_3d_data/nuwro_fake_data/merge_xs.root")
        f_wiener = uproot.open("numuCC_3d_data/nuwro_fake_data/wiener_all.root")
    elif data_type == "GENIE_v2":
        f_merged = uproot.open("numuCC_3d_data/genie_v2_fake_data/merge_xs.root")
        f_wiener = uproot.open("numuCC_3d_data/genie_v2_fake_data/wiener.root")

    print("setting event weights")

    if data_type == "real":
        data_pots = [
            1.42319e+20,
            2.5413e+20,
            2.40466e+20
        ]
    elif data_type == "NuWro":
        data_pots = [
            0,
            2.98217e+20,
            3.12922e+20
        ]
    elif data_type == "GENIE_v2":
        data_pots = [
            7.2432440e20, 
            0., 
            0.
        ]

    ext_pots = [
        2.21814e+20,
        6.25014e+20,
        7.4127e+20,
    ]

    if data_type == "real":
        include_ext = True
        include_dirt = True
    else:
        include_ext = False
        include_dirt = False


    weight_cv_vals = all_df["weight_cv"].to_numpy()
    weight_spline_vals = all_df["weight_spline"].to_numpy()
    files = all_df["file"].to_numpy()
    net_weight_vals = []
    for i in range(len(weight_cv_vals)):
        w_cv = weight_cv_vals[i]
        if not (0 < w_cv < 30):
            w_cv = 1
        
        if files[i] == "nu_overlay_run1":
            net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[0] / nu_overlay_run1_pot)
        elif files[i] == "nu_overlay_run2":
            net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[1] / nu_overlay_run2_pot)
        elif files[i] == "nu_overlay_run3":
            net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[2] / nu_overlay_run3_pot)

        elif files[i] == "dirt_run1":
            if include_dirt:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[0] / dirt_run1_pot)
            else:
                net_weight_vals.append(0)
        elif files[i] == "dirt_run2":
            if include_dirt:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[1] / dirt_run2_pot)
            else:
                net_weight_vals.append(0)
        elif files[i] == "dirt_run3":
            if include_dirt:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[2] / dirt_run3_pot)
            else:
                net_weight_vals.append(0)

        if files[i] == "ext_run1":
            if include_ext:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[0] / ext_pots[0])
            else:
                net_weight_vals.append(0)
        elif files[i] == "ext_run2":
            if include_ext:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[1] / ext_pots[1])
            else:
                net_weight_vals.append(0)
        elif files[i] == "ext_run3":
            if include_ext:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[2] / ext_pots[2])
            else:
                net_weight_vals.append(0)
        
    all_df["net_weight"] = net_weight_vals

    weight_cv_vals = all_vars_df["weight_cv"].to_numpy()
    weight_spline_vals = all_vars_df["weight_spline"].to_numpy()
    files = all_vars_df["file"].to_numpy()
    net_weight_vals = []
    for i in range(len(weight_cv_vals)):
        w_cv = weight_cv_vals[i]
        if not (0 < w_cv < 30):
            w_cv = 1
        if files[i] == "nu_overlay_run1":
            net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[0] / nu_overlay_run1_vars_pot)
        elif files[i] == "nu_overlay_run2":
            net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[1] / nu_overlay_run2_vars_pot)
        elif files[i] == "nu_overlay_run3":
            net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[2] / nu_overlay_run3_vars_pot)
        elif files[i] == "dirt_run1":
            if include_dirt:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[0] / dirt_run1_vars_pot)
            else:
                net_weight_vals.append(0)
        elif files[i] == "dirt_run2":
            if include_dirt:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[1] / dirt_run2_vars_pot)
            else:
                net_weight_vals.append(0)
        elif files[i] == "dirt_run3":
            if include_dirt:
                net_weight_vals.append(w_cv * weight_spline_vals[i] * data_pots[2] / dirt_run3_vars_pot)
            else:
                net_weight_vals.append(0)
        
    all_vars_df["net_weight"] = net_weight_vals


    print("applying the selection")

    selected_df = all_df.query("numu_cc_flag >= 0 and numu_score > 0.9 and nue_score < 7 and reco_muon_momentum>0")
    selected_vars_df = all_vars_df.query("numu_cc_flag >= 0 and numu_score > 0.9 and nue_score < 7 and reco_muon_momentum>0")


    print("getting CV selected histogram")

    reco_hist = []

    for containment in ["FC", "PC"]:
        
        if containment == "FC":
            containment_df = selected_df.query("match_isFC==1")
        else:
            containment_df = selected_df.query("match_isFC==0")
            
        for Enu_bin in range(4):
            
            if Enu_bin == 0:
                Enu_df = containment_df.query("200 < kine_reco_Enu <= 705")
            elif Enu_bin == 1:
                Enu_df = containment_df.query("705 < kine_reco_Enu < 1050")
            elif Enu_bin == 2:
                Enu_df = containment_df.query("1050 < kine_reco_Enu < 1570")
            elif Enu_bin == 3:
                Enu_df = containment_df.query("1570 < kine_reco_Enu < 4000")
            
            for theta_bin in range(9):
                
                if theta_bin == 0:
                    theta_df = Enu_df.query("-1 < reco_costheta <= -0.5")
                elif theta_bin == 1:
                    theta_df = Enu_df.query("-0.5 < reco_costheta <= 0.")
                elif theta_bin == 2:
                    theta_df = Enu_df.query("0. < reco_costheta <= 0.27")
                elif theta_bin == 3:
                    theta_df = Enu_df.query("0.27 < reco_costheta <= 0.45")
                elif theta_bin == 4:
                    theta_df = Enu_df.query("0.45 < reco_costheta <= 0.62")
                elif theta_bin == 5:
                    theta_df = Enu_df.query("0.62 < reco_costheta <= 0.76")
                elif theta_bin == 6:
                    theta_df = Enu_df.query("0.76 < reco_costheta <= 0.86")
                elif theta_bin == 7:
                    theta_df = Enu_df.query("0.86 < reco_costheta <= 0.94")
                else:
                    theta_df = Enu_df.query("0.94 < reco_costheta <= 1.")
                
                reco_hist += list(np.histogram(theta_df["reco_muon_momentum"].to_numpy(), 
                                            weights=theta_df["net_weight"].to_numpy(),
                                            bins = [i*100 for i in range(16)] + [1e9] # fifteen bins from 0 to 1500 plus an overflow
                                            )[0])
                

    print("getting data and prediction from 3D XS extraction files")
    tot_pred = []
    data = []
    if data_type == "GENIE_v2": # seems like the genie v2 root files don't contain EXT blocks, should be fine since fake data never includes EXT
        for i in range(72):
            #mc_sig_pred += list(f_merged[f"histo_{i+1}"].values(flow=True)[1:])
            #mc_bkg_pred += list(f_merged[f"histo_{i+1 + 72}"].values(flow=True)[1:])
            #ext_pred += [0 for _ in list(f_merged[f"histo_{i+1 + 1}"].values(flow=True)[1:])]
            tot_pred += list(f_merged[f"histo_{i+1}"].values(flow=True)[1:] + f_merged[f"histo_{i+1 + 72}"].values(flow=True)[1:])
            #tot_pred_from_hmc += list(f_merged[f"hmc_obsch_{i+1}"].values(flow=True)[1:])
            data += list(f_merged[f"hdata_obsch_{i+1}"].values(flow=True)[1:])
    else:
        for i in range(72):
            #mc_sig_pred += list(f_merged[f"histo_{i+1}"].values(flow=True)[1:])
            #mc_bkg_pred += list(f_merged[f"histo_{i+1 + 72}"].values(flow=True)[1:])
            #ext_pred += list(f_merged[f"histo_{i+1 + 2 * 72}"].values(flow=True)[1:])
            tot_pred += list(f_merged[f"histo_{i+1}"].values(flow=True)[1:] + f_merged[f"histo_{i+1 + 72}"].values(flow=True)[1:] + f_merged[f"histo_{i+1 + 2 * 72}"].values(flow=True)[1:])
            #tot_pred_from_hmc += list(f_merged[f"hmc_obsch_{i+1}"].values(flow=True)[1:])
            data += list(f_merged[f"hdata_obsch_{i+1}"].values(flow=True)[1:])

    print("calculating selected XS variation histograms")

    xs_cv_reco_hist = []
    universe_reco_hists = [[] for _ in range(600)]

    unisim_reco_hist_dic = {}
    for unisim_type in ["AxFFCCQEshape_UBGenie",
                                    "DecayAngMEC_UBGenie",
                                    "NormCCCOH_UBGenie",
                                    "NormNCCOH_UBGenie",
                                    "RPA_CCQE_UBGenie",
                                    "ThetaDelta2NRad_UBGenie",
                                    "Theta_Delta2Npi_UBGenie",
                                    "VecFFCCQEshape_UBGenie",
                                    "XSecShape_CCMEC_UBGenie",
                                    "xsr_scc_Fa3_SCC",
                                    "xsr_scc_Fv3_SCC"]:

        unisim_reco_hist_dic[unisim_type] = [[] for _ in range(num_unisim_variations_dic[unisim_type])]

    muon_momentum_bins = [i*100 for i in range(16)] + [1e9] # fifteen bins from 0 to 1500 plus an overflow

    for containment in ["FC", "PC"]:
        if containment == "FC":
            containment_df = selected_vars_df.query("match_isFC==1")
        else:
            containment_df = selected_vars_df.query("match_isFC==0")
        for Enu_bin in range(4):
            if Enu_bin == 0:
                Enu_df = containment_df.query("200 < kine_reco_Enu <= 705")
            elif Enu_bin == 1:
                Enu_df = containment_df.query("705 < kine_reco_Enu < 1050")
            elif Enu_bin == 2:
                Enu_df = containment_df.query("1050 < kine_reco_Enu < 1570")
            elif Enu_bin == 3:
                Enu_df = containment_df.query("1570 < kine_reco_Enu < 4000")
            for theta_bin in range(9):
                if theta_bin == 0:
                    theta_df = Enu_df.query("-1 < reco_costheta <= -0.5")
                elif theta_bin == 1:
                    theta_df = Enu_df.query("-0.5 < reco_costheta <= 0.")
                elif theta_bin == 2:
                    theta_df = Enu_df.query("0. < reco_costheta <= 0.27")
                elif theta_bin == 3:
                    theta_df = Enu_df.query("0.27 < reco_costheta <= 0.45")
                elif theta_bin == 4:
                    theta_df = Enu_df.query("0.45 < reco_costheta <= 0.62")
                elif theta_bin == 5:
                    theta_df = Enu_df.query("0.62 < reco_costheta <= 0.76")
                elif theta_bin == 6:
                    theta_df = Enu_df.query("0.76 < reco_costheta <= 0.86")
                elif theta_bin == 7:
                    theta_df = Enu_df.query("0.86 < reco_costheta <= 0.94")
                else:
                    theta_df = Enu_df.query("0.94 < reco_costheta <= 1.")

                curr_slice_cv = np.histogram(theta_df["reco_muon_momentum"].to_numpy(), weights=theta_df["net_weight"].to_numpy(), bins=muon_momentum_bins)[0]
                xs_cv_reco_hist += list(curr_slice_cv)

                for i in range(600):
                    curr_All_UBGenie_weights = [_[i] for _ in theta_df["All_UBGenie"].to_numpy()]
                    rel_weight_diffs = curr_All_UBGenie_weights / theta_df["weight_cv"].to_numpy()
                    # https://github.com/BNLIF/wcp-uboone-bdt/blob/main/src/mcm_2.h#L262-L264
                    rel_weight_diffs = np.where(np.abs(rel_weight_diffs) > 100, 1, rel_weight_diffs)
                    rel_weight_diffs = np.nan_to_num(rel_weight_diffs, nan=0)
                    curr_slice_uni = np.histogram(theta_df["reco_muon_momentum"].to_numpy(), weights=theta_df["net_weight"].to_numpy()*rel_weight_diffs, bins=muon_momentum_bins)[0]
                    universe_reco_hists[i] += list(curr_slice_uni)

                for unisim_type in ["AxFFCCQEshape_UBGenie",
                                    "DecayAngMEC_UBGenie",
                                    "NormCCCOH_UBGenie",
                                    "NormNCCOH_UBGenie",
                                    "RPA_CCQE_UBGenie",
                                    "ThetaDelta2NRad_UBGenie",
                                    "Theta_Delta2Npi_UBGenie",
                                    "VecFFCCQEshape_UBGenie",
                                    "XSecShape_CCMEC_UBGenie",
                                    "xsr_scc_Fa3_SCC",
                                    "xsr_scc_Fv3_SCC",]:

                    num_unisim_variations = num_unisim_variations_dic[unisim_type]
                    for j in range(num_unisim_variations):    

                        curr_weights = [_[j] for _ in theta_df[unisim_type].to_numpy()]
                        if not(unisim_type == "xsr_scc_Fa3_SCC" or unisim_type == "xsr_scc_Fv3_SCC"):
                            rel_weight_diffs = curr_weights / theta_df["weight_cv"].to_numpy()

                        # https://github.com/BNLIF/wcp-uboone-bdt/blob/main/src/mcm_2.h#L262-L264
                        rel_weight_diffs = np.where(np.abs(rel_weight_diffs) > 100, 1, rel_weight_diffs)
                        rel_weight_diffs = np.nan_to_num(rel_weight_diffs, nan=0)

                        curr_unisim_slice = np.histogram(theta_df["reco_muon_momentum"].to_numpy(), weights=theta_df["net_weight"].to_numpy()*rel_weight_diffs, bins=muon_momentum_bins)[0]
                        unisim_reco_hist_dic[unisim_type][j] += list(curr_unisim_slice)

    #uncollapsed_dim = len(universe_reco_hists[0])

    if collapse_type == "2D":

        print("collapsing to 2D, muon momentum and muon angle")

        # collapsing everything to muon momentum and muon angle
        # Combining FC/PC and Enu bins

        collapsed_reco_hist = np.zeros(16*9)
        collapsed_tot_pred = np.zeros(16*9)
        collapsed_data = np.zeros(16*9)

        for i in range(1152):
            collapsed_reco_hist[i%(16*9)] += reco_hist[i]
            collapsed_tot_pred[i%(16*9)] += tot_pred[i]
            collapsed_data[i%(16*9)] += data[i]

        collapsed_universe_reco_hists = []
        for uni_i in range(600):
            vals = np.zeros(16*9)
            for i in range(1152):
                vals[i%(16*9)] += universe_reco_hists[uni_i][i]
            collapsed_universe_reco_hists.append(vals)

        collapsed_unisim_reco_hist_dic = {}
        for k, v in unisim_reco_hist_dic.items():
            if k not in collapsed_unisim_reco_hist_dic:
                collapsed_unisim_reco_hist_dic[k] = []
            for uni_i in range(len(v)):
                vals = np.zeros(16*9)
                for i in range(1152):
                    vals[i%(16*9)] += v[uni_i][i]
                collapsed_unisim_reco_hist_dic[k].append(vals)


        reco_hist = collapsed_reco_hist
        tot_pred = collapsed_tot_pred
        data = collapsed_data

        universe_reco_hists = collapsed_universe_reco_hists
        unisim_reco_hist_dic = collapsed_unisim_reco_hist_dic

    elif collapse_type == "1D":

        print("collapsing to 1D, muon momentum")

        # collapsing everything to muon momentum
        # Combining FC/PC, Enu, and theta bins

        collapsed_reco_hist = np.zeros(16)
        collapsed_tot_pred = np.zeros(16)
        collapsed_data = np.zeros(16)

        for i in range(1152):
            collapsed_reco_hist[i%16] += reco_hist[i]
            collapsed_tot_pred[i%16] += tot_pred[i]
            collapsed_data[i%16] += data[i]

        collapsed_universe_reco_hists = []
        for uni_i in range(600):
            vals = np.zeros(16)
            for i in range(1152):
                vals[i%16] += universe_reco_hists[uni_i][i]
            collapsed_universe_reco_hists.append(vals)

        collapsed_unisim_reco_hist_dic = {}
        for k, v in unisim_reco_hist_dic.items():
            if k not in collapsed_unisim_reco_hist_dic:
                collapsed_unisim_reco_hist_dic[k] = []
            for uni_i in range(len(v)):
                vals = np.zeros(16)
                for i in range(1152):
                    vals[i%16] += v[uni_i][i]
                collapsed_unisim_reco_hist_dic[k].append(vals)

        reco_hist = collapsed_reco_hist
        tot_pred = collapsed_tot_pred
        data = collapsed_data

        universe_reco_hists = collapsed_universe_reco_hists
        unisim_reco_hist_dic = collapsed_unisim_reco_hist_dic

    collapsed_dim = len(universe_reco_hists[0])
    collapsed_plus_dim = collapsed_dim + 3


    print("adding M_A, MEC, and Lambda to CV and variation histograms")

    MA_values = np.genfromtxt("knob_values/MaCCQE_univs.txt")
    MEC_values = np.genfromtxt("knob_values/NormCCMEC_univs_v2.txt")
    lambda_values = [np.sum(universe_reco_hists[i]) / np.sum(reco_hist) for i in range(600)]

    tot_pred_MA = list(tot_pred) + [1.10, 1.66, 1]

    universe_reco_MAs = []
    if shape_type == "rate+shape" or shape_type == "+100":
        for i in range(600):
            universe_reco_MAs.append(np.array(list(universe_reco_hists[i]) + [MA_values[i], MEC_values[i], lambda_values[i]]))
    elif shape_type == "matrix_breakdown":
        for uni_i in range(600):
            universe_reco_MAs.append(np.array(list(universe_reco_hists[uni_i] / np.sum(universe_reco_hists[uni_i])) + [MA_values[uni_i], MEC_values[i], lambda_values[uni_i]]))
        not_normed_tot_pred_MA = tot_pred_MA.copy()
        tot_pred_MA = list(tot_pred_MA[:-3] / np.sum(tot_pred_MA[:-3])) + [1.10, 1.66, 1]
        data = data / np.sum(data)


    print("creating multisim covariance matrix")

    dim = np.array(tot_pred_MA).shape[0]

    multisim_xs_MA_cov = np.zeros((dim, dim))

    for uni_i in range(600):
        uni_reco_MA = universe_reco_MAs[uni_i]
        row_diffs = np.tile(uni_reco_MA - tot_pred_MA, (dim, 1))
        col_diffs = np.tile(np.reshape(uni_reco_MA - tot_pred_MA, (dim, 1)), (1, dim))
        multisim_xs_MA_cov += row_diffs * col_diffs

    multisim_xs_MA_cov = multisim_xs_MA_cov / 600.


    print("creating unisim covariance matrix")

    # this is fixing the fact that some of these two-length arrays contain the CV rather than a variation
    # so really we want to divide by one and not two in that case
    unisim_divide_number_dic = {
        "AxFFCCQEshape_UBGenie": 1,
        "DecayAngMEC_UBGenie": 1,
        "NormCCCOH_UBGenie": 1,
        "NormNCCOH_UBGenie": 1,
        "RPA_CCQE_UBGenie": 2,
        "ThetaDelta2NRad_UBGenie": 1,
        "Theta_Delta2Npi_UBGenie": 1,
        "VecFFCCQEshape_UBGenie": 1,
        "XSecShape_CCMEC_UBGenie": 1,
        "xsr_scc_Fa3_SCC": 10,
        "xsr_scc_Fv3_SCC": 10,
    }

    unisim_xs_MA_cov = np.zeros((collapsed_plus_dim, collapsed_plus_dim))

    if skip_AxFFCCQEshape_UBGenie:
        unisim_types = ["DecayAngMEC_UBGenie",
                        "NormCCCOH_UBGenie",
                        "NormNCCOH_UBGenie",
                        "RPA_CCQE_UBGenie",
                        "ThetaDelta2NRad_UBGenie",
                        "Theta_Delta2Npi_UBGenie",
                        "VecFFCCQEshape_UBGenie",
                        "XSecShape_CCMEC_UBGenie",
                        "xsr_scc_Fa3_SCC",
                        "xsr_scc_Fv3_SCC",]
    else:
        unisim_types = ["AxFFCCQEshape_UBGenie",
                        "DecayAngMEC_UBGenie",
                        "NormCCCOH_UBGenie",
                        "NormNCCOH_UBGenie",
                        "RPA_CCQE_UBGenie",
                        "ThetaDelta2NRad_UBGenie",
                        "Theta_Delta2Npi_UBGenie",
                        "VecFFCCQEshape_UBGenie",
                        "XSecShape_CCMEC_UBGenie",
                        "xsr_scc_Fa3_SCC",
                        "xsr_scc_Fv3_SCC",]

    unisim_diag_errs = {}
    for unisim_type in unisim_types:
        curr_unisim_xs_MA_cov = np.zeros((collapsed_plus_dim, collapsed_plus_dim))
        for j in range(num_unisim_variations_dic[unisim_type]):
            diff = np.array(unisim_reco_hist_dic[unisim_type][j]) - np.array(reco_hist)
            diff = np.append(diff, np.array([0, 0, 0]))
            row_diffs = np.tile(diff, (collapsed_plus_dim, 1))
            col_diffs = np.tile(np.reshape(diff, (collapsed_plus_dim, 1)), (1, collapsed_plus_dim))
            curr_unisim_xs_MA_cov += row_diffs * col_diffs
        curr_unisim_xs_MA_cov = curr_unisim_xs_MA_cov / unisim_divide_number_dic[unisim_type]
        unisim_xs_MA_cov += curr_unisim_xs_MA_cov
        unisim_diag_errs[unisim_type] = np.sqrt(np.diag(curr_unisim_xs_MA_cov))
    xs_MA_cov = multisim_xs_MA_cov + unisim_xs_MA_cov


    print("creating Pearson data statistical covariance matrix")

    # using Pearson data stat uncertainty
    pearson_data_stat_cov_matrix = np.zeros((collapsed_dim, collapsed_dim))
    for i in range(collapsed_dim):
        for j in range(collapsed_dim):
            if i == j:
                pearson_data_stat_cov_matrix[i][j] = tot_pred[i]
    cov_data_stat_new = pearson_data_stat_cov_matrix


    print("loading other systematic uncertainties from 3D XS extraction files")

    #cov_stat = f_wiener["hcov_stat"].to_numpy()[0]
    cov_mcstat = f_wiener["hcov_mcstat"].to_numpy()[0]
    cov_add = f_wiener["hcov_add"].to_numpy()[0]
    cov_det = f_wiener["hcov_det"].to_numpy()[0]
    cov_flux = f_wiener["hcov_flux"].to_numpy()[0]
    #cov_xs = f_wiener["hcov_xs"].to_numpy()[0]
    #cov_tot = f_wiener["hcov_tot"].to_numpy()[0]

    if collapse_type == "2D":

        collapsing_matrix = [[1] + [0 for _ in range(16*9-1)]]
        for i in range(1152):
            if i == 0:
                continue
            collapsing_matrix.append([0 for _ in range(i%(16*9))] + [1] + [0 for _ in range(16*9 - i%(16*9) - 1)])
        collapsing_matrix = np.array(collapsing_matrix)

        #cov_stat = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_stat, collapsing_matrix])
        cov_mcstat = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_mcstat, collapsing_matrix])
        cov_add = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_add, collapsing_matrix])
        cov_det = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_det, collapsing_matrix])
        cov_flux = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_flux, collapsing_matrix])

        cov_stat_MA = np.append(np.append(cov_data_stat_new, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_mcstat_MA = np.append(np.append(cov_mcstat, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_add_MA = np.append(np.append(cov_add, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_det_MA = np.append(np.append(cov_det, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_flux_MA = np.append(np.append(cov_flux, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)


    elif collapse_type == "1D":

        collapsing_matrix = [[1] + [0 for _ in range(16-1)]]
        for i in range(1152):
            if i == 0:
                continue
            collapsing_matrix.append([0 for _ in range(i%16)] + [1] + [0 for _ in range(16 - i%16 - 1)])
        collapsing_matrix = np.array(collapsing_matrix)

        #cov_stat = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_stat, collapsing_matrix])
        cov_mcstat = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_mcstat, collapsing_matrix])
        cov_add = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_add, collapsing_matrix])
        cov_det = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_det, collapsing_matrix])
        cov_flux = np.linalg.multi_dot([np.transpose(collapsing_matrix), cov_flux, collapsing_matrix])

        cov_stat_MA = np.append(np.append(cov_data_stat_new, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_mcstat_MA = np.append(np.append(cov_mcstat, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_add_MA = np.append(np.append(cov_add, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_det_MA = np.append(np.append(cov_det, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_flux_MA = np.append(np.append(cov_flux, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)

    else:
        cov_stat_MA = np.append(np.append(cov_data_stat_new, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_mcstat_MA = np.append(np.append(cov_mcstat, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_add_MA = np.append(np.append(cov_add, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_det_MA = np.append(np.append(cov_det, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)
        cov_flux_MA = np.append(np.append(cov_flux, np.zeros((3, collapsed_dim)), axis=0), np.zeros((collapsed_dim+3,3)), axis=1)


    print("combining all covariance matrices")

    total_cov_MA = (xs_MA_cov
            + cov_stat_MA
            + cov_mcstat_MA
            + cov_add_MA
            + cov_det_MA
            + cov_flux_MA
            )
    
    
    n = total_cov_MA.shape[0]

    if shape_type == "+100":

        print("adding 100% normalization error")

        percent_normalization_error = 100.

        dim = 1152

        row_diffs = np.tile(tot_pred, (dim, 1))
        col_diffs = np.tile(np.reshape(tot_pred, (dim, 1)), (1, dim))
        extra_normalization_cov = row_diffs * col_diffs
        extra_normalization_cov = np.append(
            np.append(
                extra_normalization_cov, np.zeros((3, dim)),
            axis=0), 
            np.zeros((dim+3,3))
        , axis=1)

        total_cov_MA += extra_normalization_cov * percent_normalization_error**2 / (100. * 100.)
        multisim_xs_MA_cov += extra_normalization_cov * percent_normalization_error**2 / (100. * 100.)

    elif shape_type == "matrix_breakdown":

        print("Extracting only the shape component of the XS covariance matrix")

        # non xs cov_MA
        M = (cov_stat_MA
        + cov_mcstat_MA
        + cov_add_MA
        + cov_det_MA
        + cov_flux_MA)


        # from docDB 5926

        M_s = np.zeros((n, n))
        M_n = np.zeros((n, n))
        M_m = np.zeros((n, n))

        N = np.array(not_normed_tot_pred_MA)
        N_T = np.sum(N)
        row_sums = [np.sum(M[i, :]) for i in range(n)]
        normalized_N = N / N_T
        M_sum = np.sum(M)

        print("extracting non-XS covariance matrix normalization component...")
        for i in range(n):
            for j in range(n):

                M_s[i][j] = (M[i][j]
                    - normalized_N[j] * row_sums[i]
                    - normalized_N[i] * row_sums[j]
                    + normalized_N[i] * normalized_N[j] * M_sum
                    ) / (N_T * N_T)

                M_m[i][j] = (normalized_N[j] * row_sums[i]
                    + normalized_N[i] * row_sums[j]
                    - 2. * normalized_N[i] * normalized_N[j] * M_sum
                    ) / (N_T * N_T)

                M_n[i][j] = normalized_N[i] * normalized_N[j] * M_sum / (N_T * N_T)


        total_cov_MA = M_s + xs_MA_cov

    print(f"Saving results to cache file: {cache_key}")
    with open("trio_caches/" + cache_key, 'wb') as f:
        pickle.dump((total_cov_MA, tot_pred_MA, data, multisim_xs_MA_cov, universe_reco_MAs), f)
        
    return total_cov_MA, tot_pred_MA, data, multisim_xs_MA_cov, universe_reco_MAs


def extract_trio(cov_MA, pred_MA, data, inv_cov_constraining=None):

    # data can be a array of data points, or an array of arrays for multiple data points

    trio_prior = pred_MA[-3:] # M_A, NormCCMEC, Lambda

    cov_cross = cov_MA[-3:, :-3]
    cov_prior = cov_MA[-3:, -3:]

    if inv_cov_constraining is None:
        cov_constraining = cov_MA[:-3, :-3]
        inv_cov_constraining = np.linalg.inv(cov_constraining)

    # check if the first element of data is an array, if so evaluate for each data array, but with saved covariance matrix input for speed
    if isinstance(data[0], np.ndarray):
        ret = []
        for data_i in data:
            ret.append(extract_trio(cov_MA, pred_MA, data_i, inv_cov_constraining=inv_cov_constraining))
        return ret

    constrained_trio = pred_MA[-3:] + np.linalg.multi_dot(
        [cov_cross, inv_cov_constraining, np.array(data) - np.array(pred_MA[:-3])]
    )
    constrained_trio_cov = cov_MA[-3:,-3:] - np.linalg.multi_dot(
        [cov_cross, inv_cov_constraining, np.transpose(cov_cross)]
    )

    x = constrained_trio[0]
    dx = np.sqrt(constrained_trio_cov[0][0])
    p = trio_prior[0]
    dp = np.sqrt(cov_prior[0][0])
    prior_removed_MA = (dp * dp * x - dx * dx * p) / (dp * dp - dx * dx)
    prior_removed_MA_sigma = (dp * dx) / np.sqrt(dp * dp - dx * dx)

    return constrained_trio, constrained_trio_cov, prior_removed_MA, prior_removed_MA_sigma


