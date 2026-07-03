import os

# The project root directory can be modified depending on the storage setting.
PROJECTS_ROOTDIR = '..'

# Settings of the DISFA dataset
DISFA_ROOTDIR = os.path.join(PROJECTS_ROOTDIR, 'FER_datasets/DISFA')
DISFA_IMG_ROOTDIR_LEFTCAM = os.path.join(DISFA_ROOTDIR, 'Images_LeftCamera')
DISFA_IMG_ROOTDIR_RIGHTCAM = os.path.join(DISFA_ROOTDIR, 'Images_RightCamera')
DISFA_LABEL_ROOTDIR = os.path.join(DISFA_ROOTDIR, 'ActionUnit_Labels')
DISFA_SUBJECTS = os.listdir(DISFA_LABEL_ROOTDIR)
DISFA_AUS_IE = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
DISFA_AUS_D = [1, 2, 4, 6, 9, 12, 25, 26]
DISFA_FPS = 20

# Follows the same 3-fold split as a series of previous studies using DISFA for evaluating AU recognition, such as https://github.com/ZhiwenShao/JAANet and https://github.com/ZhiwenShao/ARL
DISFA_FOLD1 = ['SN001', 'SN002', 'SN009', 'SN010', 'SN016', 'SN026', 'SN027', 'SN030', 'SN032']
DISFA_FOLD2 = ['SN006', 'SN011', 'SN012', 'SN013', 'SN018', 'SN021', 'SN024', 'SN028', 'SN031']
DISFA_FOLD3 = ['SN003', 'SN004', 'SN005', 'SN007', 'SN008', 'SN017', 'SN023', 'SN025', 'SN029']
DISFA_3FOLDS = [DISFA_FOLD1, DISFA_FOLD2, DISFA_FOLD3]

# Manually selected frames to be used as reference images
DISFA_BASELINE_FRAMES = {'SN001': 440,
                         'SN002': 5,
                         'SN003': 5,
                         'SN004': 115,
                         'SN005': 10,
                         'SN006': 1,
                         'SN007': 370,
                         'SN008': 20,
                         'SN009': 7,
                         'SN010': 1534,
                         'SN011': 90,
                         'SN012': 894,
                         'SN013': 5,
                         'SN016': 1080,
                         'SN017': 15,
                         'SN018': 5,
                         'SN021': 5,
                         'SN023': 380,
                         'SN024': 3479,
                         'SN025': 375,
                         'SN026': 5,
                         'SN027': 5,
                         'SN028': 5,
                         'SN029': 1,
                         'SN030': 30,
                         'SN031': 70,
                         'SN032': 2615}

                         

# Settings of the DISFA+ dataset
DISFAPLUS_ROOTDIR = os.path.join(PROJECTS_ROOTDIR, 'FER_datasets/DISFAPlus')
DISFAPLUS_IMG_ROOTDIR = os.path.join(DISFAPLUS_ROOTDIR, 'Images')
DISFAPLUS_LABEL_ROOTDIR = os.path.join(DISFAPLUS_ROOTDIR, 'Labels')
DISFAPLUS_SUBJECTS = [d for d in os.listdir(DISFAPLUS_LABEL_ROOTDIR) if os.path.isdir(os.path.join(DISFAPLUS_LABEL_ROOTDIR, d))]
DISFAPLUS_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

# Leave-one-participant-out 9-fold split
DISFAPLUS_FOLD1 = ['SN001']
DISFAPLUS_FOLD2 = ['SN003']
DISFAPLUS_FOLD3 = ['SN004']
DISFAPLUS_FOLD4 = ['SN007']
DISFAPLUS_FOLD5 = ['SN009']
DISFAPLUS_FOLD6 = ['SN010']
DISFAPLUS_FOLD7 = ['SN013']
DISFAPLUS_FOLD8 = ['SN025']
DISFAPLUS_FOLD9 = ['SN027']
DISFAPLUS_9FOLDS = [DISFAPLUS_FOLD1, DISFAPLUS_FOLD2, DISFAPLUS_FOLD3, DISFAPLUS_FOLD4, DISFAPLUS_FOLD5, DISFAPLUS_FOLD6, DISFAPLUS_FOLD7, DISFAPLUS_FOLD8, DISFAPLUS_FOLD9]

# Manually selected frames to be used as reference images
DISFAPLUS_BASELINE_FRAMES = {'SN001': 'A3_AU1_2_TrailNo_2/000',
                             'SN003': 'A1_AU1_TrailNo_2/000',
                             'SN004': 'A5_AU1_4b_TrailNo_1/000',
                             'SN007': 'A1_AU1_TrailNo_2/000',
                             'SN009': 'A1_AU1_TrailNo_2/000',
                             'SN010': 'A1_AU1_TrailNo_2/000',
                             'SN013': 'A1_AU1_TrailNo_2/000',
                             'SN025': 'A9_AU4_5x_TrailNo_1/000',
                             'SN027': 'A1_AU1_TrailNo_3/000'}

# Settings of the UNBC-McMaster dataset
UNBCMCMASTER_ROOTDIR = os.path.join(PROJECTS_ROOTDIR, 'FER_datasets/UNBCMcMaster')
UNBCMCMASTER_IMG_ROOTDIR = os.path.join(UNBCMCMASTER_ROOTDIR, 'Images')
UNBCMCMASTER_LABEL_ROOTDIR = os.path.join(UNBCMCMASTER_ROOTDIR, 'Frame_Labels/FACS')
UNBCMCMASTER_SUBJECTS = os.listdir(UNBCMCMASTER_IMG_ROOTDIR)
UNBCMCMASTER_AUS_IE = [4, 6, 7, 9, 10, 12, 20, 25, 26]
UNBCMCMASTER_AUS_D = [4, 6, 7, 9, 10, 12, 20, 25, 26, 43]
AU_EYE_CLOSURE = 43

#5-fold split, balancing the number of videos for each split of participants
UNBCMCMASTER_FOLD1 = ['042-ll042', '052-dr052', '092-ch092', '103-jk103', '115-jy115']
UNBCMCMASTER_FOLD2 = ['043-jh043', '059-fn059', '095-tv095', '106-nm106', '120-kz120']
UNBCMCMASTER_FOLD3 = ['047-jl047', '064-ak064', '096-bg096', '107-hs107', '121-vw121']
UNBCMCMASTER_FOLD4 = ['048-aa048', '066-mg066', '097-gf097', '108-th108', '123-jh123']
UNBCMCMASTER_FOLD5 = ['049-bm049', '080-bn080', '101-mg101', '109-ib109', '124-dn124']
UNBCMCMASTER_5FOLDS = [UNBCMCMASTER_FOLD1, UNBCMCMASTER_FOLD2, UNBCMCMASTER_FOLD3, UNBCMCMASTER_FOLD4, UNBCMCMASTER_FOLD5]

# Manually selected frames to be used as reference images
UNBCMCMASTER_BASELINE_FRAMES = {'042-ll042': 'll042t1aaaff/ll042t1aaaff001',
                                '043-jh043': 'jh043t1afaff/jh043t1afaff001',
                                '047-jl047': 'jl047t1aaaff/jl047t1aaaff001',
                                '048-aa048': 'aa048t2aaunaff/aa048t2aaunaff016',
                                '049-bm049': 'bm049t1aaaff/bm049t1aaaff001',
                                '052-dr052': 'dr052t1afaff/dr052t1afaff001',
                                '059-fn059': 'fn059t2afunaff/fn059t2afunaff013',
                                '064-ak064': 'ak064t1aaaff/ak064t1aaaff001',
                                '066-mg066': 'mg066t1aaaff/mg066t1aaaff001',
                                '080-bn080': 'bn080t1aaaff/bn080t1aaaff001',
                                '092-ch092': 'ch092t2afaff/ch092t2afaff001',
                                '095-tv095': 'tv095t2aeunaff/tv095t2aeunaff008',
                                '096-bg096': 'bg096t1aeunaff/bg096t1aeunaff008',
                                '097-gf097': 'gf097t2aiunaff/gf097t2aiunaff001',
                                '101-mg101': 'mg101t2aaunaff/mg101t2aaunaff001',
                                '103-jk103': 'jk103t1aaaff/jk103t1aaaff001',
                                '106-nm106': 'nm106t1afaff/nm106t1afaff001',
                                '107-hs107': 'hs107t2aiunaff/hs107t2aiunaff001',
                                '108-th108': 'th108t1aaaff/th108t1aaaff001',
                                '109-ib109': 'ib109t1aaaff/ib109t1aaaff001',
                                '115-jy115': 'jy115t1aeaff/jy115t1aeaff001',
                                '120-kz120': 'kz120t2afunaff/kz120t2afunaff001',
                                '121-vw121': 'vw121t1aeaff/vw121t1aeaff001',
                                '123-jh123': 'jh123t1afaff/jh123t1afaff001',
                                '124-dn124': 'dn124t1aeaff/dn124t1aeaff001'}