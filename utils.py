import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_notebook(frames, interval=50, cmap='viridis', figsize=(8, 6), 
                     colorbar=True, title_prefix='Frame', vmin=0, vmax=1):
    """
    Anima uma lista de frames (imagens) no Jupyter Notebook.
    
    Parameters:
    -----------
    frames : list ou array
        Lista de frames (arrays 2D) para animar
    interval : int, optional
        Tempo entre frames em milissegundos (default: 50)
    cmap : str, optional
        Colormap do matplotlib (default: 'viridis')
    figsize : tuple, optional
        Tamanho da figura (width, height) (default: (8, 6))
    colorbar : bool, optional
        Se True, adiciona colorbar (default: True)
    title_prefix : str, optional
        Prefixo para o título de cada frame (default: 'Frame')
    
    Returns:
    --------
    IPython.display.HTML
        Objeto HTML com a animação
    
    Example:
    --------
    >>> frames = [np.random.rand(100, 100) for _ in range(50)]
    >>> animate_notebook(frames, interval=100)
    """
    
    # Configurar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar primeiro frame
    im = ax.imshow(frames[0], cmap=cmap, animated=True, 
                   vmin=vmin, vmax=vmax, origin='lower')
    
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    ax.grid(False)
    ax.set_title(f'{title_prefix} 0')
    
    # Função de atualização
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'{title_prefix} {frame_idx}')
        return [im]
    
    # Criar animação
    ani = FuncAnimation(fig, update, frames=len(frames), 
                       interval=interval, blit=True, repeat=True)
    
    plt.close(fig)  # Evita plotar a figura estática
    
    # Retornar HTML
    return HTML(ani.to_jshtml())

from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import math

def grid_training(function,config, var_modify, initial_value, final_value, step, optimizer_class=None):
    
    list_config = []

    num_process = int(math.floor((final_value-initial_value)/step)) + 1

    for i in range(num_process):

        temp = {
        "n_epochs": config['n_epochs'],
        "batch_size": config['batch_size'],
        "model_learning_rate": config['model_learning_rate'],
        "mask_learning_rate": config['mask_learning_rate'],
        "lambda_init": config['lambda_init'],
        "lambda_factor": config['lambda_factor'],
        "lambda_patience": config['lambda_patience'],
        "lambda_treshold": config['lambda_treshold'],
        "training_id": config['training_id'] + "_gdt_" + var_modify + str(round(initial_value + i * step,6))
        # Descriptive name for the training run
        }

        temp[var_modify] = initial_value + i * step

        list_config.append(temp)

    print(list_config)

    with ProcessPoolExecutor(max_workers=num_process) as executor:

        futures = []

        for config_proc in list_config:
            future = executor.submit(function, config_proc, optimizer_class)
            futures.append(future)
            
        for future in as_completed(futures):
            print(f"Processo finalizado: {future.result()}")

    print("Finished all the processes.")



