import os, gc, sys, logging, warnings
import pandas as pd
import numpy as np
import psutil
from multiprocessing import Process, Queue
from src.data_utils import load_and_normalize_dataset
from src.aggregators import AGG_FUNCS, PAA_reduce
from src.models import get_classifier_instance, train_and_evaluate_classifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("classification.log"), logging.StreamHandler(sys.stdout)]
)

DATASETS = [
    # 'ACSF1',
    # 'CinCECGTorso',
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
clf_names = [
    "1NN-DTW", 
    "Rocket", 
    "QUANT", 
    "LITE"
]
SEED = 42
OUTPUT = 'results/results_classification.csv'

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

# --- FUNÇÃO TRABALHADORA (ISOLADA) ---
def worker_proc(clf_name, seed, X_train, y_train, X_test, y_test, queue):
    """Essa função roda em um processo separado. Quando ela acaba, a RAM morre com ela."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        from src.models import get_classifier_instance, train_and_evaluate_classifier
        
        clf = get_classifier_instance(clf_name, seed)
            
        acc, duration = train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test)
        queue.put((acc, duration)) # Envia o resultado de volta
    except Exception as e:
        queue.put(e) # Envia o erro se algo falhar

def run():
    logging.info(f"🚀 Iniciando V4 (Isolamento). RAM Base: {get_ram_usage():.2f} GB")
    os.makedirs('results', exist_ok=True)

    for i, ds_name in enumerate(DATASETS):
        logging.info(f"[{i+1}/{len(DATASETS)}] Dataset: {ds_name} | RAM: {get_ram_usage():.2f} GB")
        try:
            X_train, y_train, X_test, y_test = load_and_normalize_dataset(ds_name)

            for rate in [1.0] + RATES:
                is_baseline = (rate == 1.0)
                operators = [None] if is_baseline else list(AGG_FUNCS.keys())
                
                for op in operators:
                    # Redução PAA
                    if is_baseline:
                        X_tr_proc, X_te_proc = X_train, X_test
                    else:
                        w = int(X_train.shape[2] * rate)
                        X_tr_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_train], dtype=np.float32)
                        X_te_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_test], dtype=np.float32)

                    for clf_name in clf_names:
                        logging.info(f"   > {clf_name} | Rate: {rate} | Op: {op} | RAM: {get_ram_usage():.2f} GB")
                        
                        # --- EXECUÇÃO ISOLADA ---
                        q = Queue()
                        p = Process(target=worker_proc, args=(clf_name, SEED, X_tr_proc, y_train, X_te_proc, y_test, q))
                        p.start()
                        
                        result = q.get() # Espera o resultado ou erro
                        p.join() # O PROCESSO MORRE AQUI (Libera RAM instantaneamente)

                        if isinstance(result, Exception):
                            logging.error(f"      Erro no {clf_name}: {result}")
                        else:
                            acc, duration = result
                            res = {
                                'dataset': ds_name, 'aggregation_operator': op,
                                'retention_rate': rate, 'classifier': clf_name,
                                'accuracy': acc, 'train_test_time': duration
                            }
                            pd.DataFrame([res]).to_csv(OUTPUT, mode='a', header=not os.path.exists(OUTPUT), index=False)
                            logging.info(f"      Sucesso: Acc={acc:.4f} em {duration}s")
                        
                        # Limpeza leve no processo pai
                        gc.collect()

                    if not is_baseline: del X_tr_proc, X_te_proc
                    gc.collect()
            
            del X_train, y_train, X_test, y_test
            gc.collect()
            
        except Exception as e:
            logging.error(f"Erro Crítico no dataset {ds_name}: {str(e)}")

if __name__ == "__main__": 
    run()