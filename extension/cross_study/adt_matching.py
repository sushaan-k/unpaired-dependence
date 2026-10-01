"""Curated antibody-target matching across the three CITE-seq panels.

Each panel's antibody name is mapped to a canonical target (CD number where one
exists). Isotype controls and ambiguous targets map to None. Matching is by
target specificity only; clones differ between panels and are not matched.

Excluded as ambiguous:
  Stephenson AB_TCR_Vg2      (label reads gamma-2; the other panels carry delta-2)
  Stephenson AB_ITGAL / Hao CD11a/CD18   (Hao clone m24 is conformation-specific LFA-1)
  Stephenson AB_C5AR1 vs C5L2 (different receptors; never matched)
Hao targets measured with two clones: the clone annotated as human-specific is
used when only one is; otherwise the clone suffixed _1. For CD8 the Hao clone
named like Stephenson's antibody ("CD8") is used.
"""

STEPHENSON = {
    "AB_CD80": "CD80", "AB_CD86": "CD86", "AB_CD274": "CD274", "AB_PDCD1LG2": "CD273",
    "AB_ICOSLG": "CD275", "AB_ITGAM": "CD11b", "AB_OX40L": "CD252", "AB_TNFSF9": "CD137L",
    "AB_PVR": "CD155", "AB_NECTIN2": "CD112", "AB_CD47": "CD47", "AB_CD70": "CD70",
    "AB_TNFRSF8": "CD30", "AB_CD40": "CD40", "AB_CD40LG": "CD154", "AB_CD52": "CD52",
    "AB_CD3": "CD3", "AB_CD8": "CD8", "AB_CD56": "CD56", "AB_CD19": "CD19", "AB_CD33": "CD33",
    "AB_ITGAX": "CD11c", "AB_CD34": "CD34", "AB_TNFRSF17": "CD269", "AB_HLA-ABC": "HLA-ABC",
    "AB_THY1": "CD90", "AB_KIT": "CD117", "AB_MME": "CD10", "AB_CD45RA": "CD45RA",
    "AB_CD123": "CD123", "AB_CD7": "CD7", "AB_ITGA6": "CD49f", "AB_CCR4": "CD194",
    "AB_CD4": "CD4", "AB_CD44": "CD44", "AB_CD14": "CD14", "AB_CD16": "CD16", "AB_CD25": "CD25",
    "AB_CD45RO": "CD45RO", "AB_PD1": "CD279", "AB_TIGIT": "TIGIT",
    "AB_Mouse IgG1_K_Iso": None, "AB_Mouse_IgG2a_K_Iso": None, "AB_Mouse_IgG2b_K_Iso": None,
    "AB_Rat_IgG2b_K_Iso": None, "AB_CD20": "CD20", "AB_NCR1": "CD335", "AB_PTGDR2": "CD294",
    "AB_EPCAM": "CD326", "AB_PECAM1": "CD31", "AB_podoplanin": "Podoplanin", "AB_MCAM": "CD146",
    "AB_CDH1": "CD324", "AB_IgM": "IgM", "AB_CD5": "CD5", "AB_TCRg_d": "TCRgd",
    "AB_CXCR3": "CD183", "AB_CCR5": "CD195", "AB_FCGR2A": "CD32", "AB_CCR6": "CD196",
    "AB_CXCR5": "CD185", "AB_ITGAE": "CD103", "AB_CD69": "CD69", "AB_CD62L": "CD62L",
    "AB_CCR7": "CD197", "AB_CD161": "CD161", "AB_CTLA4": "CD152", "AB_LAG3": "CD223",
    "AB_KLRG1": "KLRG1", "AB_CD27": "CD27", "AB_LAMP1": "CD107a", "AB_FAS": "CD95",
    "AB_HLA-DR": "HLA-DR", "AB_CD1C": "CD1c", "AB_CD64": "CD64", "AB_CD141": "CD141",
    "AB_CD1d": "CD1d", "AB_KLRK1": "CD314", "AB_CEACAM8": "CD66b", "AB_CR1": "CD35",
    "AB_B3GAT1": "CD57", "AB_HAVCR2": "CD366", "AB_BTLA": "CD272", "AB_ICOS": "CD278",
    "AB_CD58": "CD58", "AB_CD96": "CD96", "AB_ENTPD1": "CD39", "AB_FASLG": "CD178",
    "AB_CX3CR1": "CX3CR1", "AB_CD24": "CD24", "AB_CD21": "CD21", "AB_ITGAL": None,
    "AB_IgA": "IgA", "AB_CD79b": "CD79b", "AB_CEACAM1/5/6": "CD66ace", "AB_CD244": "CD244",
    "AB_CD235ab": "CD235ab", "AB_MMR": "CD206", "AB_SIGLEC1": "CD169", "AB_CLEC9A": "CD370",
    "AB_XCR1": "XCR1", "AB_ITGB7": "ITGB7", "AB_BAFFR": "CD268", "AB_ICAM1": "CD54",
    "AB_SELP": "CD62P", "AB_TCR": "TCRab", "AB_VCAM1": "CD106", "AB_IL2RB": "CD122",
    "AB_TACI": "CD267", "AB_FcERIa": "FceRIa", "AB_ITGA2B": "CD41", "AB_TNFRSF9": "CD137",
    "AB_RANKL": "CD254", "AB_CD163": "CD163", "AB_CD83": "CD83", "AB_GITR": "CD357",
    "AB_KDR": "CD309", "AB_IL4R": "CD124", "AB_CXCR4": "CD184", "AB_CD2": "CD2",
    "AB_CD226": "CD226", "AB_ITGB1": "CD29", "AB_CD303": "CD303", "AB_ITGA2": "CD49b",
    "AB_CD81": "CD81", "AB_SLC3A2": "CD98", "AB_IgG_Fc": "IgGFc", "AB_IgD": "IgD",
    "AB_ITGB2": "CD18", "AB_CD28": "CD28", "AB_TSLPR": "TSLPR", "AB_CD38": "CD38",
    "AB_IL7R": "CD127", "AB_CD45": "CD45", "AB_CD15": "CD15", "AB_CD22": "CD22",
    "AB_CD71": "CD71", "AB_B7-H4": "B7-H4", "AB_DPP4": "CD26", "AB_CCR3": "CD193",
    "AB_MSR1": "CD204", "AB_CDH5": "CD144", "AB_CD1a": "CD1a", "AB_CD304": "CD304",
    "AB_CD36": "CD36", "AB_CD158": "CD158", "AB_langerin": "CD207", "AB_ITGA4": "CD49d",
    "AB_NT5E": "CD73", "AB_TCR_Va7.2": "TCRVa7.2", "AB_TCR_Vg2": None, "AB_TCR_Vg9": "TCRVg9",
    "AB_TCR_Va24-Ja18": "TCRVa24Ja18", "AB_LAIR1": "CD305", "AB_OLR1": "LOX-1",
    "AB_CD158b": "CD158b", "AB_PROM1": "CD133", "AB_CD209": "CD209", "AB_KIR3DL1": "CD158e1",
    "AB_KIR2DL5A": "CD158f", "AB_NCR3": "CD337", "AB_NCR2": "CD336", "AB_FCRL4": "CD307d",
    "AB_FCRL5": "CD307e", "AB_SLAMF7": "CD319", "AB_SDC1": "CD138", "AB_CD99": "CD99",
    "AB_CLEC12A": "CD371", "AB_phosphoTau": "phosphoTau", "AB_BAFF": "BAFF", "AB_KLRD1": "CD94",
    "AB_SLAMF1": "CD150", "AB_Igkappa": "Igkappa", "AB_LGALS3": "Galectin-3",
    "AB_LILRB1": "CD85j", "AB_FCER2": "CD23", "AB_Iglambda": "Iglambda", "AB_HLA-A_2": "HLA-A2",
    "AB_LRRC32": "GARP", "AB_SIGLEC7": "CD328", "AB_TCR_VB_13_1": "TCRVb13.1", "AB_CD82": "CD82",
    "AB_CD101": "CD101", "AB_IL21R": "CD360", "AB_C5AR1": "CD88", "AB_HLA-F": "HLA-F",
    "AB_NLRP2": "NLRP2", "AB_Podocalyxin": "Podocalyxin", "AB_GGT1": "GGT1", "AB_c-Met": "c-Met",
    "AB_LIGHT": "CD258", "AB_DR3": "DR3",
}

COLON = {
    "B7-H4": "B7-H4", "C5L2": "C5L2", "CD10": "CD10", "CD102": "CD102", "CD103": "CD103",
    "CD105": "CD105", "CD107a--LAMP-1": "CD107a", "CD112--Nectin-2": "CD112",
    "CD115--CSF-1R": "CD115", "CD117--c-kit": "CD117", "CD119--IFN-G-R-A-chain": "CD119",
    "CD11b": "CD11b", "CD11c": "CD11c", "CD120b": "CD120b", "CD123": "CD123",
    "CD127--IL-7Ra": "CD127", "CD134--OX40": "CD134", "CD135": "CD135", "CD137": "CD137",
    "CD137L--4-1BB-Ligand": "CD137L", "CD138--Syndecan-1": "CD138", "CD14.2": "CD14",
    "CD140a": "CD140a", "CD140b": "CD140b", "CD141--Thrombomodulin": "CD141",
    "CD144--VE-cadherin": "CD144", "CD146": "CD146", "CD15--BEA-1": "CD15", "CD152": "CD152",
    "CD154": "CD154", "CD155--PVR": "CD155", "CD158--KIR2DL1--KIR2DS1--KIR2DS3--KIR2DS5": "CD158",
    "CD158b--KIR2DL2--KIR2DL3--NKAT2": "CD158b", "CD158e1--KIR3DL1--NKB1": "CD158e1",
    "CD158f--KIR2DL5": "CD158f", "CD16": "CD16", "CD161": "CD161", "CD163.1": "CD163",
    "CD169--Sialoadhesin--Siglec-1": "CD169", "CD172a--SIRPa": "CD172a", "CD178--FasL": "CD178",
    "CD18": "CD18", "CD183--CXCR3": "CD183", "CD184--CXCR4": "CD184", "CD185--CXCR5": "CD185",
    "CD186--CXCR6": "CD186", "CD19.1": "CD19", "CD192--CCR2": "CD192", "CD193--CCR3": "CD193",
    "CD194--CCR4": "CD194", "CD195--CCR5": "CD195", "CD196--CCR6": "CD196",
    "CD197--CCR7": "CD197", "CD1a": "CD1a", "CD1c": "CD1c", "CD1d": "CD1d", "CD2.1": "CD2",
    "CD20": "CD20", "CD202b--Tie2--Tek": "CD202b", "CD21": "CD21", "CD22.1": "CD22",
    "CD223--LAG-3": "CD223", "CD226--DNAM-1.1": "CD226", "CD23": "CD23", "CD24.1": "CD24",
    "CD244--2B4": "CD244", "CD25": "CD25", "CD253--TRAIL": "CD253",
    "CD254--TRANCE--RANKL": "CD254", "CD26": "CD26", "CD267--TACI": "CD267",
    "CD268--BAFF-R": "CD268", "CD269--BCMA": "CD269", "CD27.1": "CD27",
    "CD270--HVEM--TR2": "CD270", "CD272--BTLA": "CD272", "CD273--B7-DC--PD-L2": "CD273",
    "CD274--B7-H1--PD-L1": "CD274", "CD275--B7-H2--ICOSL": "CD275", "CD276--B7-H3": "CD276",
    "CD278--ICOS": "CD278", "CD279--PD-1": "CD279", "CD28.1": "CD28", "CD284--TLR4": "CD284",
    "CD29": "CD29", "CD294--CRTH2": "CD294", "CD3": "CD3", "CD30": "CD30",
    "CD301--CLEC10A": "CD301", "CD303--BDCA-2": "CD303", "CD304--Neuropilin-1": "CD304",
    "CD305--LAIR1": "CD305", "CD31": "CD31", "CD314--NKG2D": "CD314", "CD319--CRACC": "CD319",
    "CD32": "CD32", "CD325--N-Cadherin": "CD325", "CD326--EpCAM": "CD326",
    "CD328--Siglec-7": "CD328", "CD33.1": "CD33", "CD335--NKp46": "CD335",
    "CD336--NKp44": "CD336", "CD337--NKp30": "CD337", "CD338--ABCG2": "CD338",
    "CD34.1": "CD34", "CD35": "CD35", "CD357--GITR": "CD357", "CD36.1": "CD36",
    "CD366--Tim-3": "CD366", "CD370--CLEC9A--DNGR1": "CD370", "CD371--CLEC12A": "CD371",
    "CD38.1": "CD38", "CD39": "CD39", "CD4.2": "CD4", "CD40.1": "CD40", "CD41": "CD41",
    "CD44.1": "CD44", "CD45": "CD45", "CD45RA": "CD45RA", "CD45RO": "CD45RO", "CD47.1": "CD47",
    "CD48.1": "CD48", "CD49a": "CD49a", "CD49b": "CD49b", "CD49d": "CD49d", "CD49f": "CD49f",
    "CD5.1": "CD5", "CD52.1": "CD52", "CD54": "CD54", "CD55.1": "CD55", "CD56--NCAM": "CD56",
    "CD57": "CD57", "CD58.1": "CD58", "CD59.1": "CD59", "CD61": "CD61", "CD62L": "CD62L",
    "CD63.1": "CD63", "CD64": "CD64", "CD69.1": "CD69", "CD7.1": "CD7", "CD70.1": "CD70",
    "CD71": "CD71", "CD72.1": "CD72", "CD73--Ecto-5-nucleotidase": "CD73",
    "CD79b--Ig-B": "CD79b", "CD80.1": "CD80", "CD81--TAPA-1": "CD81", "CD83.1": "CD83",
    "CD85g--ILT7": "CD85g", "CD85j--ILT2": "CD85j", "CD86.1": "CD86", "CD8a": "CD8",
    "CD9.1": "CD9", "CD90--Thy1": "CD90", "CD93.1": "CD93", "CD94": "CD94", "CD95--Fas": "CD95",
    "CD96--TACTILE": "CD96", "CD98": "CD98", "CX3CR1.1": "CX3CR1", "FceRIa": "FceRIa",
    "HLA-DR": "HLA-DR", "IgD": "IgD", "IgG-Fc": "IgGFc", "IgM": "IgM", "KLRG1--MAFA": "KLRG1",
    "MERTK.1": "MERTK", "TCR-A--TCR-B": "TCRab", "TCR-G--TCR-D": "TCRgd",
    "TCR-V-A-7.2": "TCRVa7.2", "TCR-V-A-24-J-A-18--iNKT cell": "TCRVa24Ja18",
    "TCR-V-D-2": "TCRVd2", "TCR-V-G-9": "TCRVg9", "TIGIT--VSTM3": "TIGIT", "Tim-4": "TIM-4",
    "TSLPR": "TSLPR", "XCR1.1": "XCR1",
}

# Hao et al. 2021 3' panel (GEO feature names; the spreadsheet writes _ for -).
HAO = {
    "CD39": "CD39", "Rat-IgG1-1": None, "CD107a": "CD107a", "CD62P": "CD62P",
    "TCR-2": "TCRab", "CD30": "CD30", "CD31": "CD31", "CD34": "CD34", "CD35": "CD35",
    "CD36": "CD36", "CD223": "CD223", "TIGIT": "TIGIT", "TCR-V-9": "TCRVg9", "CD226": "CD226",
    "CD178": "CD178", "CD319": "CD319", "CD171": "CD171", "Siglec-8": "Siglec-8",
    "CD340": "CD340", "Rat-IgG2b": None, "VEGFR-3": "VEGFR-3", "CD29": "CD29",
    "CD62E": "CD62E", "CD4-2": None, "CD4-1": "CD4", "CD22": "CD22", "CD3-1": "CD3",
    "CD20": "CD20", "CD27": "CD27", "CD45RB": "CD45RB", "CD25": "CD25", "CD24": "CD24",
    "CD146": "CD146", "Galectin-9": "Galectin-9", "CD142": "CD142", "CD141": "CD141",
    "CD294": "CD294", "Rat-IgG1-2": None, "CD45RA": "CD45RA", "CX3CR1": "CX3CR1",
    "CD56-2": None, "CD56-1": "CD56", "CD45RO": "CD45RO", "CD303": "CD303", "GP130": "GP130",
    "CD253": "CD253", "CD357": "CD357", "CD11b-1": None, "CD354": "CD354", "CD11b-2": "CD11b",
    "CLEC12A": "CD371", "CD38-2": None, "CD38-1": "CD38", "Folate": "FRbeta",
    "Rag-IgG2c": None, "CD209": "CD209", "CD152": "CD152", "CD154": "CD154", "CD155": "CD155",
    "Cadherin": "Cadherin-11", "CD201": "CD201", "CD204": "CD204", "CD205": "CD205",
    "CD206": "CD206", "CD207": "CD207", "CD1d": "CD1d", "CD284": "CD284", "CD1c": "CD1c",
    "Podoplanin": "Podoplanin", "CD1a": "CD1a", "CD366": "CD366", "IgD": "IgD", "IgM": "IgM",
    "CD66a/c/e": "CD66ace", "CD49d": "CD49d", "LOX-1": "LOX-1", "TIM-4": "TIM-4",
    "CD98": "CD98", "CD370": "CD370", "CD49a": "CD49a", "CD44-2": "CD44", "C5L2": "C5L2",
    "CD44-1": None, "CD158e1": "CD158e1", "CD124": "CD124", "CD127": "CD127",
    "CD126": "CD126", "CD279": "CD279", "CD278": "CD278", "CD123": "CD123", "CD122": "CD122",
    "CD96": "CD96", "CD274": "CD274", "CD95": "CD95", "CD271": "CD271", "CD270": "CD270",
    "CD90": "CD90", "CD272": "CD272", "CD16": "CD16", "CD14": "CD14", "CD15": "CD15",
    "CD13": "CD13", "CD267": "CD267", "CD26-2": None, "CD200": "CD200", "CD26-1": "CD26",
    "CD18": "CD18", "CD19": "CD19", "CD194": "CD194", "TCR-1": "TCRgd",
    "TCR-V-7.2": "TCRVa7.2", "CD70": "CD70", "CD71": "CD71", "CD72": "CD72", "CD73": "CD73",
    "TCR-V-2": "TCRVd2", "CD177": "CD177", "CD301": "CD301", "CD140a": "CD140a",
    "CD140b": "CD140b", "CD305": "CD305", "CD304": "CD304", "CD2": "CD2", "CD309": "CD309",
    "CD85g": "CD85g", "CD110": "CD110", "CD8": "CD8", "CD9": "CD9", "HLA-DR": "HLA-DR",
    "CD137": "CD137", "CD134": "CD134", "CD135": "CD135", "CD61": "CD61", "CD192": "CD192",
    "CD268": "CD268", "CD269": "CD269", "CD81": "CD81", "CD80": "CD80", "CD83": "CD83",
    "CD193": "CD193", "TSLPR": "TSLPR", "CD86": "CD86", "CCR10": "CCR10",
    "Notch-1": "Notch-1", "Notch-2": "Notch-3", "CD337": "CD337", "CD79b": "CD79b",
    "CD275-2": None, "CD275-1": "CD275", "CD79a": "CD79a", "CD49b": "CD49b", "CD64": "CD64",
    "CD63": "CD63", "CD45-2": None, "CD45-1": "CD45", "CD133-2": None, "CD133-1": "CD133",
    "CD69": "CD69", "CD68": "CD68", "CD314": "CD314", "CD11a/CD18": None, "CD186": "CD186",
    "CD185": "CD185", "CD184": "CD184", "CD103": "CD103", "CD102": "CD102", "CD106": "CD106",
    "CD105": "CD105", "CD66b": "CD66b", "CD252": "CD252", "CD109": "CD109",
    "TCR-V-24-J-18": "TCRVa24Ja18", "Integrin-7": "ITGB7", "CD158b": "CD158b",
    "CD158f": "CD158f", "CD8a": None, "CD203c": "CD203c", "CD52": "CD52", "CD195": "CD195",
    "CD196": "CD196", "CD57": "CD57", "CD54": "CD54", "CD55": "CD55", "CD99": "CD99",
    "CD59": "CD59", "CD199": "CD199", "CD93": "CD93", "CD244": "CD244", "CD158": "CD158",
    "CD235ab": "CD235ab", "CD273": "CD273", "CD243": "CD243", "CD325": "CD325",
    "CD324": "CD324", "CD307e": "CD307e", "CD172a": "CD172a", "CD307d": "CD307d",
    "CD42b": "CD42b", "CD115": "CD115", "CD117": "CD117", "XCR1": "XCR1", "CD112": "CD112",
    "MERTK": "MERTK", "B7-H4": "B7-H4", "CD21": "CD21", "CD307c/FcRL3": "CD307c",
    "CLEC2": "CLEC2", "CD48": "CD48", "CD47": "CD47", "CD46": "CD46", "CD41": "CD41",
    "CD40": "CD40", "CD43": "CD43", "CD338": "CD338", "CD235a": "CD235a", "CD335": "CD335",
    "CD3-2": None, "CD119": "CD119", "CD169": "CD169", "CD28": "CD28", "CD161": "CD161",
    "CD163": "CD163", "CD138-1": "CD138", "CD164": "CD164", "CD138-2": None, "CD144": "CD144",
    "CD202b": "CD202b", "CD11c": "CD11c",
}


def matched_targets(*panels):
    """Canonical targets present (and not excluded) in every panel, sorted."""
    sets = [{v for v in p.values() if v is not None} for p in panels]
    return sorted(set.intersection(*sets))


def names_for(panel, targets):
    inverse = {}
    for name, target in panel.items():
        if target is not None:
            assert target not in inverse, (target, name, inverse.get(target))
            inverse[target] = name
    return [inverse[t] for t in targets]
