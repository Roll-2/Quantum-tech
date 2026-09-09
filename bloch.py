import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Costanti
DEG_TO_RAD = np.pi / 180
INCREMENTO = 15 * DEG_TO_RAD  # 15 gradi in radianti

# Creazione figura e assi 3D
fig = plt.figure("Bloch sphere", figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Angolo di visuale fisso
ax.view_init(elev=30, azim=45)

# Disegno sfera (wireframe)
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color='lightblue', alpha=0.3, linewidth=0.5)

# Assi colorati
ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='purple', linewidth=2)
ax.text(0, 0, 1.3, 'Z', color='purple', fontsize=12)
ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='yellow', linewidth=2)
ax.text(1.5, 0, 0, 'X', color='yellow', fontsize=12)
ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='green', linewidth=2)
ax.text(0, 1.3, 0, 'Y', color='green', fontsize=12)

# Variabili globali per gli angoli
theta = 0.0  # angolo verticale (0 = |0>, π = |1>)
phi = 0.0    # angolo orizzontale (fase)
freccia = ax.quiver(0, 0, 0, 0, 0, 1, color='red', arrow_length_ratio=0.1, linewidth=2)

# Testo informativo
testo = fig.text(0.75, 0.5, '', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                 verticalalignment='center')

def format_angle(rad):
    """Converte un angolo in radianti in una stringa con multipli di π."""
    fractions = [
        (0, "0"),
        (1/6, "π/6"), (1/4, "π/4"), (1/3, "π/3"), (1/2, "π/2"),
        (2/3, "2π/3"), (3/4, "3π/4"), (5/6, "5π/6"), (1, "π"),
        (7/6, "7π/6"), (5/4, "5π/4"), (4/3, "4π/3"), (3/2, "3π/2"),
        (5/3, "5π/3"), (7/4, "7π/4"), (11/6, "11π/6"), (2, "2π")
    ]
    x = rad / np.pi
    for value, name in fractions:
        if abs(x - value) < 0.01:
            return name
    return f"{x:.2f}π"

def format_coefficient(val):
    """Formatta un coefficiente (seno o coseno di mezzo angolo) in frazione se possibile."""
    # Valori notevoli per theta/2 (0, π/6, π/4, π/3, π/2)
    # Qui val è cos(theta/2) o sin(theta/2)
    # Mappiamo i valori assoluti a frazioni note
    abs_val = abs(val)
    if abs(abs_val - 1) < 1e-3:
        return "1"
    if abs(abs_val - np.sqrt(3)/2) < 1e-3:
        return "√3/2"
    if abs(abs_val - np.sqrt(2)/2) < 1e-3:
        return "√2/2"
    if abs(abs_val - 1/2) < 1e-3:
        return "1/2"
    if abs(abs_val) < 1e-3:
        return "0"
    # Altrimenti decimale
    return f"{val:.3f}"

def aggiorna_freccia(theta, phi):
    """Aggiorna la freccia e il testo."""
    global freccia
    x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    
    freccia.remove()
    freccia = ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=2)
    
    # Coefficienti in forma frazionaria quando possibile
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    
    coef0 = format_coefficient(cos_half)
    coef1 = format_coefficient(sin_half)
    
    # Gestione fase
    if abs(sin_half) > 1e-6:
        # Arrotondamento di phi ai valori notevoli per la visualizzazione
        phi_for_fase = phi
        # Se phi è vicino a 0, π, 2π, non mostra fase
        if abs(phi) < 1e-3 or abs(phi - 2*np.pi) < 1e-3:
            fase_str = ""
        elif abs(phi - np.pi) < 1e-3:
            fase_str = "e^{iπ}"
        elif abs(phi - np.pi/2) < 1e-3:
            fase_str = "e^{iπ/2}"
        elif abs(phi - 3*np.pi/2) < 1e-3:
            fase_str = "e^{i3π/2}"
        else:
            fase_str = f"e^{{{format_angle(phi)}}}"
    else:
        fase_str = ""
    
    # Costruzione stato
    if abs(sin_half) < 1e-3:
        stato = rf"$|\psi\rangle = {coef0}|0\rangle$"
    elif abs(cos_half) < 1e-3:
        stato = rf"$|\psi\rangle = {coef1}{fase_str}|1\rangle$"
    else:
        # Segno
        if sin_half > 0:
            stato = rf"$|\psi\rangle = {coef0}|0\rangle + {coef1}{fase_str}|1\rangle$"
        else:
            stato = rf"$|\psi\rangle = {coef0}|0\rangle - {abs(sin_half):.3f}{fase_str}|1\rangle$"
    
    angoli = rf"$\theta={format_angle(theta)},\ \phi={format_angle(phi)}$"
    testo.set_text(stato + "\n" + angoli)
    fig.canvas.draw_idle()

def snap_phi(phi):
    """Arrotonda phi al multiplo di π/12 più vicino tra quelli notevoli."""
    # phi = un multiplo di π/12 (15°) per avere valori precisi
    # Calcoliamo il multiplo più vicino
    step = np.pi / 12  # 15 gradi
    # Arrotonda al multiplo più vicino
    phi_snap = round(phi / step) * step
    # Mantiene in [0, 2π)
    phi_snap = phi_snap % (2 * np.pi)
    return phi_snap

def on_mouse_move(event):
    """Il mouse controlla solo θ (verticale)."""
    global theta
    if event.y is None:
        return
    altezza = fig.get_size_inches()[1] * fig.dpi
    theta = np.pi * (1 - event.y / altezza)
    theta = np.clip(theta, 0, np.pi)
    aggiorna_freccia(theta, phi)

def on_key_press(event):
    """Gestione tasti: A/D incrementano φ di 15°, X/Y/Z vanno agli assi, R reset."""
    global theta, phi
    if event.key == 'a':
        phi = (phi - INCREMENTO) % (2 * np.pi)
        phi = snap_phi(phi)  # arrotonda per precisione
        aggiorna_freccia(theta, phi)
    elif event.key == 'd':
        phi = (phi + INCREMENTO) % (2 * np.pi)
        phi = snap_phi(phi)
        aggiorna_freccia(theta, phi)
    elif event.key == 'x':
        # Stato |+> = (|0> + |1>)/√2  → theta=π/2, phi=0
        theta, phi = np.pi/2, 0.0
        aggiorna_freccia(theta, phi)
    elif event.key == 'y':
        # Stato |+i> = (|0> + i|1>)/√2 → theta=π/2, phi=π/2
        theta, phi = np.pi/2, np.pi/2
        aggiorna_freccia(theta, phi)
    elif event.key == 'z':
        # Stato |0>
        theta, phi = 0.0, 0.0
        aggiorna_freccia(theta, phi)
    elif event.key == 'r':
        # reset
        theta, phi = 0.0, 0.0
        aggiorna_freccia(theta, phi)

# Collega eventi
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('key_press_event', on_key_press)

print("CONTROLLI:")
print("- Muovi il mouse VERTICALMENTE per cambiare θ (distanza dai poli)")
print("- Premi A/D per cambiare φ di 15° (π/12)")
print("- Premi X/Y/Z per andare agli stati positivi degli assi")
print("- Premi R per resettare")

plt.show()
