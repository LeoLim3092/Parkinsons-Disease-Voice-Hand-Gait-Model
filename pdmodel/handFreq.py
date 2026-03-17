import glob
import joblib
from handFeaturesExtraction import preprocess_landmarks, get_freq_inten


path = r'../../handOutput/*_A*_hand.txt'
files_ls = glob.glob(path)

path = r'../../handOutput2/*_A*_hand.txt'
files2_ls = glob.glob(path)

path = r'../../handOutput3/*_A*_hand.txt'
files3_ls = glob.glob(path)

all_ls = files_ls + files2_ls + files3_ls

error_ls = []
path_ls = []
f_ls = []
e_ls = []
f_e_ls = []


for f in all_ls:
    try:
        dt = joblib.load(f)
        arr = preprocess_landmarks(dt)[:,0,:,:]
        f, e, f_e = get_freq_inten(arr)
        path_ls.append(f)
        f_ls.append(f)
        e_ls.append(e)
        f_e_ls.append(f_e)
        
        print("Finish {}".format(f)) 

    except:
        error_ls.append(f)

dt = {
    "name": path_ls,
    "freq": f_ls,
    "intensity": e_ls,
    "freq_inten":f_e_ls
}

joblib.dump(dt, "../../20240508_handFreq.txt")
joblib.dump(error_ls, "../../20240508_handfreq_error.txt")

