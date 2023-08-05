import os

import imageio
import numpy as np
import matplotlib.pyplot as plt


def get_fourier_coeffs(iv, N):
    coeffs = np.zeros(N)
    K = len(iv)
    # n: 1 -> N
    for n in range(1, N + 1):
        # k: 1 -> K
        for k in range(1, K + 1):
            # Megadjuk az integrálás határait
            zmin = (k - 1) * (1 / K)
            zmax = k * (1 / K)
            # Kezdeti értékek közül kiválasztjuk a k.-at
            A = iv[k - 1]
            # Integrandus definiálása
            integrand = 2 * A * np.sin(n * np.pi *
                                       np.linspace(zmin, zmax))
            # Kiszámítjuk az n. együtthatót a trapézszabály használatával
            # (az "np.linspace" fgv alapvetően 50 pontra osztja az adott
            # intervallumot, a pontosság növeléséhez 'N' mellett ezt is változtathatjuk)
            coeffs[n - 1] += np.trapz(integrand,
                                      np.linspace(zmin, zmax))
    return coeffs


def get_sol(coeffs, x_ticks, diffusivity, ts):
    # Csupa 0 mtx a megoldások tárolására
    sol = np.zeros((len(ts), len(x_ticks)))
    # Iterálunk minden időpontra
    for t in ts:
        y = np.zeros_like(x_ticks)
        # n: 1 -> N
        # (itt nem adtuk át 'N'-t, mivel pont a "coeffs" lista hossza)
        for n in range(1, len(coeffs) + 1):
            # Kiszámítjuk a megoldást a t időpontban
            y += coeffs[n - 1] * np.sin(n * np.pi * x_ticks) \
                 * np.exp(-1 * diffusivity * (n ** 2) * (np.pi ** 2) * t)
        # 0-100 intervallumra szórítjuk a hőmérsékletet
        y = np.clip(y, 0, 100)
        # Eltároljuk a t időpontbeli hőmérsékleteket
        sol[t - 1, :] = y
    return sol


def create_3dplot(x_ticks, ts, sol):
    # Rácskoordináták létrehozása, ahol kiértékeltük a függvényt
    X, t = np.meshgrid(x_ticks, ts)
    # Matplotlib objektumok inicializálása
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Függvény ábrázolás 'viridis' színsémával
    ax.plot_surface(X, t, sol, cmap='viridis')
    # A tengelyek elnevezése
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('t(x, t) in Celsius')
    # Ábra kimentése
    plt.savefig('3Dplot')
    # Ábra megjelenítése
    plt.show()
    # Ábra bezárása (ha ezt nem tesszük meg, benne marad a memóriában, és a következő ábra is rákerül)
    plt.close()


def create_animation(x_ticks, ts, sol):
    # Ha nem létezik a "frames" mappa, létrehozzuk
    os.makedirs('frames', exist_ok=True)
    images = []
    # Minden egész időpontban ábrázoljuk a megoldást a pozíció szerint,
    # majd ezeket az ábrákat egymás után fűzve kapunk egy animációt
    for t in ts:
        # A megoldás ábrázolása a t. időpillanatban
        plt.plot(x_ticks, sol[t - 1, :])
        # Ábra elnevezése
        plt.title(f'Eltelt idő = {t} mp')
        # Tengelyek elnevezése
        plt.xlabel('x')
        plt.ylabel('t(x, t) Celsius-ban')
        # Ábrázolt tartományok meghatározása
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        # Ábra kimentése, bezárása
        plt.savefig(f'frames/frame_{t}.png')
        plt.close()
        # Ábra beolvasása, hozzácsatolása az "images" listához
        image = imageio.imread(f'frames/frame_{t}.png')
        images.append(image)
    # Animáció készítése és kimentése az images listában tárolt ábrákból
    imageio.mimsave('RandomHeatSim.gif', images, duration=0.1)


def main():
    # Fourier együtthatók száma
    N = 1000
    # Diffúziós együttható
    diffusivity = 0.000111
    # Rúd felosztása
    x_ticks = np.linspace(0, 1, 100)  # 0,1 intervallumról veszünk 100 pontot egyenletesen elosztva
    # Időtartam felosztása, elemek típúsának átalakítása int-re
    # (nem tudunk listából float/double típusú változóval kiválasztani)
    ts = np.linspace(1, 60, 60).astype(int)
    # Kezdeti hőmérsékletek
    iv = np.array([88, 33, 70, 11, 3, 75, 55, 45, 90, 60])

    coeffs = get_fourier_coeffs(iv, N)
    sol = get_sol(coeffs, x_ticks, diffusivity, ts)

    create_3dplot(x_ticks, ts, sol)
    create_animation(x_ticks, ts, sol)

if __name__ == '__main__':
    main()
