import numpy as np
from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict
from sklearn.preprocessing import LabelEncoder

def get_UCR_univariate_list():
    return [
        'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
        'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
        'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
        'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
        'CricketZ', 'Crop', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
        'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
        'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
        'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
        'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
        'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
        'MoteStrain', 'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
        'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
        'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
        'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
        'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
        'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
        'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    ]

def get_UCR_multivariate_list():
    return [
        'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
        'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
        'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
        'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
        'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
        'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
        'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
        'StandWalkJump', 'UWaveGestureLibrary'
    ]

def label(y):
    label = np.unique(y)
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)
    return y

def load_articulary():
    X = np.load('csmt/datasets/data/UCR/ArticularyWordRecognition/X.npy')
    y = np.load('csmt/datasets/data/UCR/ArticularyWordRecognition/y.npy')
    y=label(y)
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_natops():
    X = np.load('csmt/datasets/data/UCR/NATOPS/X.npy')
    y = np.load('csmt/datasets/data/UCR/NATOPS/y.npy')
    y=label(y)
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_natops_raw():
    X = np.load('csmt/datasets/data/UCR/NATOPS/X.npy')
    y = np.load('csmt/datasets/data/UCR/NATOPS/y.npy')
    X_train = np.load('csmt/datasets/data/UCR/NATOPS/X_train.npy')
    y_train = np.load('csmt/datasets/data/UCR/NATOPS/y_train.npy')
    y_train=y_train.astype(float)-1

    X_test = np.load('csmt/datasets/data/UCR/NATOPS/X_valid.npy')
    y_test = np.load('csmt/datasets/data/UCR/NATOPS/y_valid.npy')
    y_test=y_test.astype(float)-1

    X_val,y_val=X_test,y_test

    X_train=X_train.astype(np.float32)
    X_test=X_test.astype(np.float32)
    X_val=X_val.astype(np.float32)
    y_train=y_train.astype(np.int32)
    y_test=y_test.astype(np.int32)
    y_val=y_val.astype(np.int32)

    mask=get_true_mask([column for column in X])
    return X_train,y_train,X_val,y_val,X_test,y_test,mask

def load_atrial():
    X = np.load('csmt/datasets/data/UCR/AtrialFibrillation/X.npy')
    y = np.load('csmt/datasets/data/UCR/AtrialFibrillation/y.npy')
    y=label(y)
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_china_town():
    X = np.load('csmt/datasets/data/UCR/Chinatown/X.npy')
    X=np.reshape(X,(X.shape[0],X.shape[2],X.shape[1]))
    y = np.load('csmt/datasets/data/UCR/Chinatown/y.npy')
    y=label(y)
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_flow_mampf():
    from csmt.classifiers.classic.Markov import get_data
    from csmt.classifiers.classic.Markov.models import SLMarkovClassify,SLFeature
    from sklearn.linear_model import LogisticRegression as LR
    import argparse
    parser = argparse.ArgumentParser('run the markov model')
    parser.add_argument('--train_json', type=str, default='csmt/datasets/data/flow_mampf/train.json', help='the processed train json file')
    parser.add_argument('--test_json', type=str, default='csmt/datasets/data/flow_mampf/test.json', help='the processed test json file')
    parser.add_argument('--status_label', type=str, default='csmt/datasets/data/flow_mampf/', help='the status label')

    # for preprocessing
    parser.add_argument('--class_num', type=int, default=20, help='the class number')
    parser.add_argument('--min_length', type=int, default=2, help='the flow under this parameter will be filtered')
    parser.add_argument('--max_packet_length', type=int, default=6000, help='the largest packet length')
    
    config = parser.parse_args()
    train, test = get_data(config)
    
    # print(type(train[3]))
    # print(train[3])

    model_fea=SLFeature()
    X_train,y_train=model_fea.get_feature(train[0], train[3])
    print(X_train)
    
    # modx=SLMarkovClassify(LR(C=1, class_weight=None), 1)
    # modx.fit(train[0], train[3])
    # pred = modx.predict(test[0], test[3])
    # print(pred)
    
    # from csmt.classifiers.classic.Markov import eval
    # res = eval.evaluate(test[2], pred)


def load_flow_fsnet():

    train_path = "csmt/datasets/data/flow_fsnet/train_pcap_length.txt" 
    train_file = open(train_path,"r").readlines()

    test_path = "csmt/datasets/data/flow_fsnet/test_pcap_length.txt" 
    test_file = open(test_path,"r").readlines()

    X_train,y_train=get_flow_Xy(train_file)
    X_test,y_test=get_flow_Xy(test_file)

    X_val,y_val=X_test,y_test

    X_train=X_train.astype(np.float32)
    X_test=X_test.astype(np.float32)
    X_val=X_val.astype(np.float32)
    y_train=y_train.astype(np.int32)
    y_test=y_test.astype(np.int32)
    y_val=y_val.astype(np.int32)

    mask=get_true_mask([column for column in X_train])

    return X_train,y_train,X_val,y_val,X_test,y_test,mask


def get_flow_Xy(train_file):
    num_train = len(train_file)
    n_steps = 256
    n_inputs = 1
    X_train= np.zeros([num_train,n_steps,n_inputs])
    y_train = np.zeros([num_train],np.int32)
    import random
    random.shuffle(train_file)
    label2idx = {"iqiyi":0,"taobao":1,"weixin":2}
    idx = 0
    for line in train_file:
        content = line.strip().split()
        y_train[idx] = label2idx[content[0].split("_")[0]]
        for i in range(1, 257):
            X_train[idx][i-1][0] = int(content[i])*1.0
        idx += 1
    return X_train,y_train

