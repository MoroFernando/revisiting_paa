import time, os, gc, sys, logging, warnings
import pandas as pd
import numpy as np
import psutil
from src.data_utils import load_and_normalize_dataset
from src.aggregators import AGG_FUNCS, PAA_reduce
from src.models import get_classifiers
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("classification.log"), logging.StreamHandler(sys.stdout)]
)

DATASETS = [
    # 'ACSF1',
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

RATES = [0.85, 0.70, 0.55, 0.40, 0.25]
SEED = 42
OUTPUT = 'results/results_classification_2try.csv'

def get_ram_usage():
    """Retorna o uso atual de RAM em GB para debug de OOM."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def run():
    logging.info(f"Iniciando Experimentos. RAM Base: {get_ram_usage():.2f} GB")
    os.makedirs('results', exist_ok=True)
    
    total_datasets = len(DATASETS)

    for i, ds_name in enumerate(DATASETS):
        logging.info(f"[{i+1}/{total_datasets}] Dataset: {ds_name} | RAM: {get_ram_usage():.2f} GB")
        
        try:
            X_train, y_train, X_test, y_test = load_and_normalize_dataset(ds_name)
            
            for rate in [1.0] + RATES:
                is_baseline = (rate == 1.0)
                operators = [None] if is_baseline else list(AGG_FUNCS.keys())
                
                for op in operators:
                    op_str = str(op)
                    
                    # Redução de Dados
                    if is_baseline:
                        X_tr_proc, X_te_proc = X_train, X_test
                    else:
                        w = int(X_train.shape[2] * rate)
                        X_tr_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_train])
                        X_te_proc = np.array([[PAA_reduce(s, w, op) for s in sample] for sample in X_test])

                    clfs_dict = get_classifiers(SEED)
                    for clf_name, clf in clfs_dict.items():

                        logging.info(f"   > Treinando: {clf_name} | Rate: {rate} | Op: {op_str} | RAM: {get_ram_usage():.2f} GB")
                        
                        try:
                            start_time = time.time()
                            clf.fit(X_tr_proc, y_train)
                            y_pred = clf.predict(X_te_proc)
                            
                            total_time = np.round(time.time() - start_time, 2)
                            acc = accuracy_score(y_test, y_pred)
                            
                            # Salvar Resultados
                            res = {
                                'dataset': ds_name, 
                                'aggregation_operator': op,
                                'retention_rate': rate, 
                                'classifier': clf_name,
                                'accuracy': acc, 
                                'train_test_time': total_time
                            }
                            pd.DataFrame([res]).to_csv(OUTPUT, mode='a', header=not os.path.exists(OUTPUT), index=False)
                            logging.info(f"      Sucesso: Acc={acc:.4f} em {total_time}s")

                        except MemoryError:
                            logging.error(f"      OOM (Falta de RAM) capturado no {clf_name}!")
                        except Exception as e:
                            logging.error(f"      Erro no {clf_name}: {str(e)}")
                        
                        # 3. Limpeza de Memória Pós-Execução
                        del clf
                        if 'tensorflow' in sys.modules:
                            import tensorflow as tf
                            tf.keras.backend.clear_session()
                        gc.collect()

                    if not is_baseline: del X_tr_proc, X_te_proc
                    gc.collect()
            
            # Limpar Dataset da RAM antes do próximo
            logging.info(f"Finalizado dataset {ds_name}. Limpando RAM...")
            del X_train, y_train, X_test, y_test
            gc.collect()
            
        except Exception as e:
            logging.error(f"Erro Crítico no dataset {ds_name}: {str(e)}")

if __name__ == "__main__": 
    run()