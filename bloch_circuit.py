import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import time

# ============================================
# Funzioni di utilità quantistica
# ============================================

def stato_da_angoli(theta, phi):
    """Restituisce il vettore di stato |ψ> = [cos(θ/2), e^(iφ) sin(θ/2)]"""
    return np.array([np.cos(theta/2), np.sin(theta/2) * np.exp(1j * phi)])

def angoli_da_stato(psi):
    """Da un vettore di stato (normalizzato) restituisce (θ, φ)"""
    a, b = psi[0], psi[1]
    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    a /= norm
    b /= norm
    # Rendi a reale positivo (ruota fase globale)
    phase = np.angle(a)
    a = np.abs(a)
    b = b * np.exp(-1j * phase)
    theta = 2 * np.arccos(np.clip(a, -1, 1))
    phi = np.angle(b) % (2*np.pi)
    return theta, phi

def applica_gate(psi, gate_matrix):
    """Applica una matrice 2x2 al vettore di stato e restituisce il nuovo stato"""
    return gate_matrix @ psi

# Definizione delle matrici dei gate
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=complex)
Sd = np.array([[1, 0], [0, -1j]], dtype=complex)  # S†
Td = np.array([[1, 0], [0, np.exp(-1j * np.pi/4)]], dtype=complex)  # T†

# Dizionario per l'animazione: asse di rotazione (normalizzato) e angolo totale
gate_axis_angle = {
    'I': (np.array([0, 0, 1]), 0.0),
    'X': (np.array([1, 0, 0]), np.pi),
    'Y': (np.array([0, 1, 0]), np.pi),
    'Z': (np.array([0, 0, 1]), np.pi),
    'H': (np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)]), np.pi),
    'S': (np.array([0, 0, 1]), np.pi/2),
    'T': (np.array([0, 0, 1]), np.pi/4),
    'S†': (np.array([0, 0, 1]), -np.pi/2),
    'T†': (np.array([0, 0, 1]), -np.pi/4),
}

def ruota_vettore(v, asse, angolo):
    """Ruota il vettore 3D 'v' attorno ad 'asse' (normalizzato) dell'angolo dato (rad)."""
    asse = asse / np.linalg.norm(asse)
    cos_a = np.cos(angolo)
    sin_a = np.sin(angolo)
    return v * cos_a + np.cross(asse, v) * sin_a + asse * np.dot(asse, v) * (1 - cos_a)

# ============================================
# Setup della figura e della sfera di Bloch
# ============================================

fig = plt.figure("Bloch Sphere with Quantum Gates", figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=45)

# Sfera wireframe
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
ax.text(1.3, 0, 0, 'X', color='yellow', fontsize=12)
ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='green', linewidth=2)
ax.text(0, 1.3, 0, 'Y', color='green', fontsize=12)

# Stato iniziale |0>
theta, phi = 0.0, 0.0
vettore = np.array([0, 0, 1.0])
freccia = ax.quiver(0, 0, 0, vettore[0], vettore[1], vettore[2],
                    color='red', arrow_length_ratio=0.1, linewidth=2)

# Testo informativo
testo = fig.text(0.75, 0.5, '', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                 verticalalignment='center')

# Variabili di stato
psi = stato_da_angoli(theta, phi)

# ============================================
# Funzioni di formattazione (frazioni e angoli)
# ============================================

def format_fraction(val, eps=1e-6):
    if abs(val) < eps:
        return "0"
    fractions = [
        (1, "1"), (1/2, "1/2"), (1/3, "1/3"), (2/3, "2/3"),
        (1/4, "1/4"), (3/4, "3/4"), (1/np.sqrt(2), "1/√2"),
        (np.sqrt(3)/2, "√3/2"), (1/np.sqrt(3), "1/√3")
    ]
    for fraction, name in fractions:
        if abs(abs(val) - fraction) < eps:
            return name
    return f"{val:.3f}"

def format_angle(rad):
    fractions = [
        (0, "0"), (1/6, "π/6"), (1/4, "π/4"), (1/3, "π/3"),
        (1/2, "π/2"), (2/3, "2π/3"), (3/4, "3π/4"), (5/6, "5π/6"),
        (1, "π"), (7/6, "7π/6"), (5/4, "5π/4"), (4/3, "4π/3"),
        (3/2, "3π/2"), (5/3, "5π/3"), (7/4, "7π/4"), (11/6, "11π/6"), (2, "2π")
    ]
    x = rad / np.pi
    for value, name in fractions:
        if abs(x - value) < 0.01:
            return name
    return f"{x:.2f}π"

# ============================================
# Funzioni di aggiornamento grafico
# ============================================

def aggiorna_freccia_da_psi(psi_nuovo):
    """Aggiorna la freccia e il testo a partire dal vettore di stato."""
    global psi, freccia, vettore, theta, phi
    psi = psi_nuovo
    theta, phi = angoli_da_stato(psi)
    vettore = np.array([np.sin(theta)*np.cos(phi),
                        np.sin(theta)*np.sin(phi),
                        np.cos(theta)])
    freccia.remove()
    freccia = ax.quiver(0, 0, 0,
                        vettore[0], vettore[1], vettore[2],
                        color='red', arrow_length_ratio=0.1, linewidth=2)

    # Costruzione testo stato
    a, b = psi[0], psi[1]
    # Rendi a reale positivo per estetica
    phase = np.angle(a)
    a = np.abs(a)
    b = b * np.exp(-1j * phase)
    coef0 = format_fraction(a)
    coef1 = format_fraction(np.abs(b))
    if np.abs(b) < 1e-6:
        stato = rf"$|\psi\rangle = {coef0}|0\rangle$"
    elif np.abs(a) < 1e-6:
        ang_fase = np.angle(b)
        if np.abs(ang_fase) < 1e-6:
            stato = rf"$|\psi\rangle = |1\rangle$"
        elif np.abs(ang_fase - np.pi) < 1e-6:
            stato = rf"$|\psi\rangle = -|1\rangle$"
        else:
            stato = rf"$|\psi\rangle = e^{{i{format_angle(ang_fase)}}}|1\rangle$"
    else:
        parte1 = f"{coef0}|0⟩"
        if np.abs(np.angle(b)) < 1e-6:
            segno = "+"
            fase_str = ""
        elif np.abs(np.angle(b) - np.pi) < 1e-6:
            segno = "-"
            fase_str = ""
        else:
            segno = "+"
            fase_str = f"e^{{i{format_angle(np.angle(b))}}}"
        stato = rf"$|\psi\rangle = {parte1} {segno} {coef1}{fase_str}|1\rangle$"

    angoli = rf"$\theta={format_angle(theta)},\ \phi={format_angle(phi)}$"
    testo.set_text(stato + "\n" + angoli)
    fig.canvas.draw_idle()

def applica_e_aggiorna(gate_matrix):
    """Applica un gate e aggiorna il display."""
    nuovo_psi = applica_gate(psi, gate_matrix)
    aggiorna_freccia_da_psi(nuovo_psi)

# ============================================
# Controlli da tastiera (frecce e A/D)
# ============================================

def on_key_press(event):
    global theta, phi
    step_theta = np.pi / 24  # passo più piccolo per θ (7.5°)
    step_phi = np.pi / 12     # 15° per φ

    if event.key == 'up':
        theta = np.clip(theta - step_theta, 0, np.pi)
        nuovo_psi = stato_da_angoli(theta, phi)
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'down':
        theta = np.clip(theta + step_theta, 0, np.pi)
        nuovo_psi = stato_da_angoli(theta, phi)
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'a':
        phi = (phi - step_phi) % (2*np.pi)
        nuovo_psi = stato_da_angoli(theta, phi)
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'd':
        phi = (phi + step_phi) % (2*np.pi)
        nuovo_psi = stato_da_angoli(theta, phi)
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'r':
        theta, phi = 0.0, 0.0
        nuovo_psi = stato_da_angoli(theta, phi)
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'x':
        nuovo_psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # |+>
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'y':
        nuovo_psi = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])  # |+i>
        aggiorna_freccia_da_psi(nuovo_psi)
    elif event.key == 'z':
        nuovo_psi = np.array([1, 0])  # |0>
        aggiorna_freccia_da_psi(nuovo_psi)

fig.canvas.mpl_connect('key_press_event', on_key_press)

# ============================================
# Widgets: bottoni per i gate (con nomi estesi e colori pastello)
# ============================================

button_color = '#e6f0fa'      # azzurrino pastello
hover_color = '#b0c4de'       # azzurro più scuro al passaggio

# Definiamo i bottoni con etichette più leggibili
gate_buttons_info = [
    ('I', I, 'Identity'),
    ('X', X, 'Pauli-X'),
    ('Y', Y, 'Pauli-Y'),
    ('Z', Z, 'Pauli-Z'),
    ('H', H, 'Hadamard'),
    ('S', S, 'Phase S'),
    ('T', T, 'T'),
    ('S†', Sd, 'S†'),
    ('T†', Td, 'T†')
]

# Posizionamento in basso
left = 0.05
bottom = 0.02
width = 0.09
height = 0.05
spacing = 0.005

buttons = []
for i, (short, mat, long_name) in enumerate(gate_buttons_info):
    ax_btn = plt.axes([left + i*(width+spacing), bottom, width, height])
    btn = Button(ax_btn, long_name, color=button_color, hovercolor=hover_color)
    btn.on_clicked(lambda event, m=mat: applica_e_aggiorna(m))
    buttons.append(btn)

# Bottone Play per la sequenza animata
ax_play = plt.axes([left + len(gate_buttons_info)*(width+spacing) + 0.02, bottom, width, height])
play_btn = Button(ax_play, 'Play', color='#c1e1c1', hovercolor='#98fb98')  # verde pastello

# ============================================
# Animazione della sequenza
# ============================================

sequenza = [('H', H), ('S', S), ('T', T)]  # H → S → T
animazione_in_corso = False
timer = None

def esegui_step_animazione(indice_step, totale_step, gate_idx, asse, angolo_totale, v_iniziale):
    """Richiamata dal timer per ogni step di rotazione."""
    global animazione_in_corso, timer, psi
    if indice_step >= totale_step:
        # Fine rotazione per questo gate, passa al prossimo
        gate_idx += 1
        if gate_idx >= len(sequenza):
            animazione_in_corso = False
            return
        nome, _ = sequenza[gate_idx]
        asse, ang_tot = gate_axis_angle[nome]
        v_iniziale = vettore.copy()
        timer = fig.canvas.new_timer(interval=50)
        timer.add_callback(esegui_step_animazione, 0, 30, gate_idx, asse, ang_tot, v_iniziale)
        timer.start()
        return

    ang_parz = angolo_totale * indice_step / totale_step
    v_ruotato = ruota_vettore(v_iniziale, asse, ang_parz)
    theta_nuovo = np.arccos(np.clip(v_ruotato[2], -1, 1))
    phi_nuovo = np.arctan2(v_ruotato[1], v_ruotato[0]) % (2*np.pi)
    nuovo_psi = stato_da_angoli(theta_nuovo, phi_nuovo)
    aggiorna_freccia_da_psi(nuovo_psi)

    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(esegui_step_animazione, indice_step+1, totale_step, gate_idx, asse, angolo_totale, v_iniziale)
    timer.start()

def play_sequenza(event):
    """Avvia l'animazione della sequenza."""
    global animazione_in_corso, timer
    if animazione_in_corso:
        return
    animazione_in_corso = True
    gate_idx = 0
    nome, _ = sequenza[gate_idx]
    asse, ang_tot = gate_axis_angle[nome]
    v_iniziale = vettore.copy()
    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(esegui_step_animazione, 0, 30, gate_idx, asse, ang_tot, v_iniziale)
    timer.start()

play_btn.on_clicked(play_sequenza)

# ============================================
# Istruzioni a console
# ============================================

print("=" * 60)
print("SFERA DI BLOCH INTERATTIVA CON GATE")
print("=" * 60)
print("CONTROLLI DA TASTIERA:")
print("  ↑/↓ : cambia θ (angolo da Z)")
print("  A/D : cambia φ (fase) di 15°")
print("  X/Y/Z: stati assiali |+>, |+i>, |0>")
print("  R    : reset a |0>")
print("\nBOTTONI:")
print("  Clicca sui gate per applicarli istantaneamente")
print("  'Play' esegue la sequenza H → S → T (animata)")
print("=" * 60)

plt.show()
