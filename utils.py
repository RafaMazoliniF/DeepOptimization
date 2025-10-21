import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_notebook(frames, interval=50, cmap='viridis', figsize=(8, 6), 
                     colorbar=True, title_prefix='Frame', vmin=0, vmax=1):
    """
    Anima uma lista de frames (imagens 2D ou 3D) no Jupyter Notebook.
    Se os frames forem 3D (com canais), cada canal será exibido em um subplot separado.
    
    Parameters:
    -----------
    frames : list ou array
        Lista de frames para animar. Cada frame pode ser um array 2D (H, W) ou 3D (C, H, W).
    interval : int, optional
        Tempo entre frames em milissegundos (default: 50)
    cmap : str, optional
        Colormap do matplotlib (default: 'viridis')
    figsize : tuple, optional
        Tamanho da figura (width, height) (default: (8, 6))
    colorbar : bool, optional
        Se True, adiciona colorbar (default: True)
    title_prefix : str, optional
        Prefixo para o título de cada animação (default: 'Frame')
    
    Returns:
    --------
    IPython.display.HTML
        Objeto HTML com a animação
    
    Example:
    --------
    >>> frames = [np.random.rand(100, 100) for _ in range(50)]
    >>> animate_notebook(frames, interval=100)
    """
    
    first_frame = np.array(frames[0])
    is_multichannel = first_frame.ndim == 3
    num_channels = first_frame.shape[0] if is_multichannel else 1

    # Ajusta o tamanho da figura para acomodar os subplots
    if is_multichannel:
        fig_width, fig_height = figsize
        figsize = (fig_width * num_channels, fig_height)

    # Configura a figura e os subplots
    fig, axes = plt.subplots(1, num_channels, figsize=figsize, squeeze=False)
    axes = axes.flatten() # Garante que axes seja sempre um array iterável
    
    # Plota o primeiro frame em cada subplot
    images = []
    for i in range(num_channels):
        ax = axes[i]
        frame_data = first_frame[i] if is_multichannel else first_frame
        im = ax.imshow(frame_data, cmap=cmap, animated=True, 
                       vmin=vmin, vmax=vmax, origin='lower')
        images.append(im)
        
        if colorbar:
            fig.colorbar(im, ax=ax)
        
        ax.grid(False)
        title = f'Channel {i} - {title_prefix} 0' if is_multichannel else f'{title_prefix} 0'
        ax.set_title(title)
    
    # Função de atualização
    def update(frame_idx):
        current_frames = np.array(frames[frame_idx])
        for i in range(num_channels):
            frame_data = current_frames[i] if is_multichannel else current_frames
            images[i].set_array(frame_data)
            title = f'Channel {i} - {title_prefix} {frame_idx}' if is_multichannel else f'{title_prefix} {frame_idx}'
            axes[i].set_title(title)
        return images
    
    # Criar animação
    ani = FuncAnimation(fig, update, frames=len(frames), 
                       interval=interval, blit=True, repeat=False)
    
    plt.close(fig)  # Evita plotar a figura estática
    
    # Retornar HTML
    return HTML(ani.to_jshtml())

from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import math

def grid_training(function,config, var_modify, initial_value, final_value, step, optimizer_class=None):
    
    list_config = []

    num_process = int(math.floor((final_value-initial_value)/step)) + 3

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
        "training_id": config['training_id'] 
        # Descriptive name for the training run
        }

        temp[var_modify] = initial_value * (step ** i)

        temp['training_id'] = config['training_id'] + "_gdt_" + var_modify + str(round(temp[var_modify],6))

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
