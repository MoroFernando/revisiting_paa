import os, gc, sys, logging, warnings
import pandas as pd
import numpy as np
import psutil
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
    """Retorna o uso atual de RAM em GB para debug de OOM."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def run():
    logging.info(f"Iniciando Experimentos. RAM Base: {get_ram_usage():.2f} GB")
    os.makedirs('results', exist_ok=True)

    for i, ds_name in enumerate(DATASETS):
        logging.info(f"[{i+1}/{len(DATASETS)}] Dataset: {ds_name} | RAM: {get_ram_usage():.2f} GB")
        
        try:
            X_train, y_train, X_test, y_test = load_and_normalize_dataset(ds_name)

            for rate in [1.0] + RATES:
                is_baseline = (rate == 1.0)
                operators = [None] if is_baseline else list(AGG_FUNCS.keys())
                
                for op in operators:
                    # Redução de Dados (PAA)
                    if is_baseline:
                        X_tr_proc, X_te_proc = X_train, X_test
                    else:
                        w = int(X_train.shape[2] * rate)
                        X_tr_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_train], dtype=np.float32)
                        X_te_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_test], dtype=np.float32)

                    # Loop de Classificadores
                    for clf_name in clf_names:
                        logging.info(f"   > {clf_name} | Rate: {rate} | Op: {op} | RAM: {get_ram_usage():.2f} GB")
                        
                        try:
                            # 1. Instanciar
                            clf = get_classifier_instance(clf_name, SEED)
                            
                            # 2. Treinar e Avaliar
                            acc, duration = train_and_evaluate_classifier(clf, X_tr_proc, y_train, X_te_proc, y_test)
                            
                            # 3. Salvar
                            res = {
                                'dataset': ds_name, 'aggregation_operator': op,
                                'retention_rate': rate, 'classifier': clf_name,
                                'accuracy': acc, 'train_test_time': duration
                            }
                            pd.DataFrame([res]).to_csv(OUTPUT, mode='a', header=not os.path.exists(OUTPUT), index=False)
                            logging.info(f"      Sucesso: Acc={acc:.4f} em {duration}s")

                        except MemoryError:
                            logging.error(f"      OOM no {clf_name}!")
                        except Exception as e:
                            logging.error(f"      Erro no {clf_name}: {str(e)}")
                        finally:
                            if 'clf' in locals(): del clf
                            if 'tensorflow' in sys.modules:
                                import tensorflow as tf
                                tf.keras.backend.clear_session()
                            gc.collect()

                    # Limpar datasets reduzidos
                    if not is_baseline: 
                        del X_tr_proc, X_te_proc
                        gc.collect()
            
            # Limpar Dataset base
            del X_train, y_train, X_test, y_test
            gc.collect()
            
        except Exception as e:
            logging.error(f"Erro Crítico no dataset {ds_name}: {str(e)}")

if __name__ == "__main__": 
    run()