import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def get_fourier_coeffs(iv, N):
    coeffs = np.zeros(N)
    for n in range(1, N + 1):
        for k in range(1, len(iv) + 1):
            # Megadjuk az integrálás határait
            zmin = (k - 1) * 0.1
            zmax = zmin + 0.1
            # Kezdeti értékek közül kiválasztjuk a k.-at
            A = iv[k - 1]
            # Integrandus definiálása
            integrand = 2 * A * np.sin(n * np.pi * np.linspace(zmin, zmax))
            # Kiszámítjuk az n. együtthatót a trapézszabály használatával
            coeffs[n - 1] += np.trapz(integrand,
                                      np.linspace(zmin, zmax))
    return coeffs


def get_sol(coeffs, x_ticks, diffusivity, ts):
    soln = np.zeros((len(ts), len(x_ticks)))
    for t in ts.astype(int):
        y = np.zeros_like(x_ticks)
        for n in range(1, len(coeffs) + 1):
            y += coeffs[n - 1] * np.sin(n * np.pi * x_ticks) * np.exp(-1 * diffusivity * (n ** 2) * (np.pi ** 2) * t)

        # 0-100 intervallumra szórítjuk a hőmérsékletet
        y = np.clip(y, 0, 100)
        # Eltároljuk a T időpontbeli hőmérsékleteket
        soln[t - 1, :] = np.flip(y)
    return soln


def create_3dplot(x_ticks, ts, sol):
    os.makedirs('plots', exist_ok=True)
    X, t = np.meshgrid(x_ticks, ts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, t, sol, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('t(x, t) in Celsius')
    plt.savefig('plots/3Dplot')
    plt.show()
    plt.close()


def create_animation(x_ticks, ts, sol):
    os.makedirs('plots/frames', exist_ok=True)
    images = []
    for t in ts.astype(int):
        # FF = np.full(len(temperature), -0.125)
        plt.plot(x_ticks, sol[t - 1, :])
        # plt.plot(FF, temperature, 'white')
        plt.title(f'Eltelt idő = {t} mp')
        plt.xlabel('x')
        plt.ylabel('t(x, t) Celsius-ban')

        plt.xlim(0, 1)
        plt.ylim(0, 100)

        plt.savefig(f'plots/frames/frame_{t}.png')
        plt.close()

        images.append(imageio.imread(f'plots/frames/frame_{t}.png'))

    imageio.mimsave('plots/RandomHeatSim.gif', images, duration=0.05)


def main():
    # Paraméterek megadása
    N = 500
    diffusivity = 0.000111
    x_ticks = np.linspace(0, 1, 100)
    # Kezdeti hőmérsékletek
    iv = np.array([88, 33, 70, 11, 3, 75, 55, 45, 90, 60])

    ts = np.linspace(1, 120, 120)

    coeffs = get_fourier_coeffs(iv, N)
    sol = get_sol(coeffs, x_ticks, diffusivity, ts)

    create_3dplot(x_ticks, ts, sol)
    create_animation(x_ticks, ts, sol)


if __name__ == '__main__':
    main()
