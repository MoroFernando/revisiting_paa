import time, os, gc, sys, logging
import pandas as pd
import numpy as np
from src.data_utils import load_and_normalize_dataset
from src.aggregators import AGG_FUNCS, PAA_reduce
from src.models import get_classifiers
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("experiment_clf_details.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DATASETS = [
  'ACSF1',
  'CinCECGTorso',
  'EOGHorizontalSignal',
  'EOGVerticalSignal',
  'EthanolLevel',
  'HandOutlines',
  'Haptics',
  'HouseTwenty',
  'InlineSkate',
  'Mallat',
  'MixedShapesRegularTrain',
  'MixedShapesSmallTrain',
  'Phoneme',
  'PigAirwayPressure',
  'PigArtPressure',
  'PigCVP',
  'Rock',
  'SemgHandGenderCh2',
  'SemgHandMovementCh2',
  'SemgHandSubjectCh2',
  'StarLightCurves',
]
RATES = [
  0.85,
  0.70, 
  0.55,
  0.40,
  0.25,
]
SEED = 42
OUTPUT = 'results/results_classification.csv'

def run():
    logging.info("Starting classification experiments...")
    os.makedirs('results', exist_ok=True)
    
    clfs_dict = get_classifiers(SEED)
    total_datasets = len(DATASETS)

    for i, ds_name in enumerate(DATASETS):
        logging.info(f"[{i+1}/{total_datasets}] Loading dataset: {ds_name}")
        
        try:
            X_train, y_train, X_test, y_test = load_and_normalize_dataset(ds_name)
            
            for rate in [1.0] + RATES:
                is_baseline = (rate == 1.0)
                operators = [None] if is_baseline else AGG_FUNCS.keys()
                
                for op in operators:
                    logging.info(f"   > Processing rate: {rate} | Operator: {op}")
                    
                    if is_baseline:
                        X_tr_proc, X_te_proc = X_train, X_test
                    else:
                        w = int(X_train.shape[2] * rate)
                        X_tr_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_train])
                        X_te_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_test])

                    for clf_name, clf in clfs_dict.items():
                        logging.info(f"      - Training classifier: {clf_name}")
                        start_time = time.time()
                        
                        clf.fit(X_tr_proc, y_train)
                        y_pred = clf.predict(X_te_proc)
                        
                        total_time = np.round(time.time() - start_time, 2)
                        acc = accuracy_score(y_test, y_pred)
                        
                        logging.info(f"      - Finished: Acc={acc:.4f} at {total_time}s")
                        
                        res = {
                            'dataset': ds_name, 
                            'aggregation_operator': op,
                            'retention_rate': rate, 
                            'classifier': clf_name,
                            'accuracy': acc, 
                            'train_test_time': total_time
                        }
                        pd.DataFrame([res]).to_csv(OUTPUT, mode='a', header=not os.path.exists(OUTPUT), index=False)
                    
                    if not is_baseline: del X_tr_proc, X_te_proc
                    gc.collect()
            
            logging.info(f"Finished dataset: {ds_name}")
            
        except Exception as e:
            logging.error(f"Error on processing {ds_name}: {str(e)}")

if __name__ == "__main__": 
    run()