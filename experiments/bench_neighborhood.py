import os, gc, sys, logging, warnings, time
import pandas as pd
import numpy as np
import psutil
from multiprocessing import Process, Queue
from src.data_utils import load_and_normalize_dataset
from src.aggregators import AGG_FUNCS, PAA_reduce
from src.metrics import calculate_preservation_at_k, calculate_trustworthiness

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("neighborhood.log"), logging.StreamHandler(sys.stdout)]
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
    0.25
]
K = 5
OUTPUT = 'results/results_neighborhood.csv'

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

def neighborhood_worker(X_orig, X_red, k, queue):
    try:
        n_samples = X_orig.shape[0]
        X_orig_flat = X_orig.reshape(n_samples, -1)
        X_red_flat = X_red.reshape(n_samples, -1)
        
        p_at_k = calculate_preservation_at_k(X_orig_flat, X_red_flat, k=k)
        trust = calculate_trustworthiness(X_orig_flat, X_red_flat, k=k)
        
        queue.put((p_at_k, trust))
    except Exception as e:
        queue.put(e)

def run_neighborhood():
    logging.info(f"Starting Neighborhood Benchmark. Base RAM: {get_ram_usage():.2f} GB")
    os.makedirs('results', exist_ok=True)

    for i, ds_name in enumerate(DATASETS):
        logging.info(f"[{i+1}/{len(DATASETS)}] Dataset: {ds_name}")
        
        try:
            X_train, _, _, _ = load_and_normalize_dataset(ds_name)
            
            # 100% Rate (Baseline)
            for rate in [1.0] + RATES:
                is_baseline = (rate == 1.0)
                operators = [None] if is_baseline else list(AGG_FUNCS.keys())
                
                for op in operators:
                    logging.info(f"   > Op: {op} | Rate: {rate} | RAM: {get_ram_usage():.2f} GB")
                    
                    # PAA Reduction
                    if is_baseline:
                        X_reduced = X_train
                    else:
                        w = int(X_train.shape[2] * rate)
                        X_reduced = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_train], dtype=np.float32)

                    # Isolate neighborhood calculations in a separate process to manage memory better
                    q = Queue()
                    p = Process(target=neighborhood_worker, args=(X_train, X_reduced, K, q))
                    p.start()
                    
                    result = q.get()
                    p.join()

                    if isinstance(result, Exception):
                        logging.error(f"      Error in calculation: {result}")
                    else:
                        p_at_k, trust = result
                        res = {
                            'dataset': ds_name,
                            'aggregation_operator': op,
                            'retention_rate': rate,
                            'precision@5': p_at_k,
                            'trustworthiness': trust
                        }
                        pd.DataFrame([res]).to_csv(OUTPUT, mode='a', header=not os.path.exists(OUTPUT), index=False)
                        logging.info(f"      P@5: {p_at_k:.4f} | Trust: {trust:.4f}")

                    if not is_baseline: del X_reduced
                    gc.collect()

            del X_train
            gc.collect()

        except Exception as e:
            logging.error(f"Critical Error in dataset {ds_name}: {str(e)}")

if __name__ == "__main__":
    run_neighborhood()