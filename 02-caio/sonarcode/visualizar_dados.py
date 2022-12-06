import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_spectrum(corrida, title):
    plt.imshow(corrida, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.show()
  
def plot_lofargram(fig, ax, lofargram_data, title):
  m_fontsize = 12
  im = ax.imshow(
      lofargram_data,
      cmap="jet",
      extent=[1, lofargram_data.shape[1], lofargram_data.shape[0], 1],
      aspect="auto"
  )

  ax.set_title(
      title,
      fontsize= m_fontsize,
      fontweight="bold"
  )

  ax.set_ylabel("Janelas", fontsize=m_fontsize, fontweight="bold")
  ax.set_xlabel("Intervalos", fontsize=m_fontsize, fontweight="bold")
  ax.set_xticks(np.linspace(0, lofargram_data.shape[1], 9))

  cbar = fig.colorbar(im, ax=ax)
  cbar.ax.set_ylabel('dB', fontweight='bold', fontsize=m_fontsize)
  
def get_lofargram(corrida, ship_label):
    fig = plt.figure(1, figsize=(10,8))
    gs1 = gridspec.GridSpec(1, 1)
    ax_list = [fig.add_subplot(ss) for ss in gs1]
    ax = ax_list[0]

    # lofargram_data = preprocessed_data[self.run][ship_label][0].copy().copy()
    lofargram_data = corrida.copy().copy()

    # plot_lofargram(fig, ax, lofargram_data, f"Lofargrama da {ship_label}")

    title = f"Lofargrama da {ship_label}"

    m_fontsize = 12
    im = ax.imshow(
      lofargram_data,
      cmap="jet",
      extent=[1, lofargram_data.shape[1], lofargram_data.shape[0], 1],
      aspect="auto"
    )

    ax.set_title(
      title,
      fontsize= m_fontsize,
      fontweight="bold"
    )

    ax.set_ylabel("Tempo (segundos)", fontsize=m_fontsize, fontweight="bold")
    ax.set_xlabel("Bins de Frequência", fontsize=m_fontsize, fontweight="bold")
    ax.set_xticks(np.linspace(0, lofargram_data.shape[1], 9))

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('dB', fontweight='bold', fontsize=m_fontsize)
    plt.savefig('/gdrive/MyDrive/lps/goltz/resultados_marlon/'+str(ship_label)+'.png', format='png')
    plt.show()

    del lofargram_data
