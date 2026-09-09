import numpy as np
import matplotlib.pyplot as plt
import random
import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate

class QuantumBattleshipGrover:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size          # 10.000
        self.n_qubits = int(np.ceil(np.log2(self.n_cells)))  # 14 qubit (16.384 stati)
        self.ship_index = random.randint(0, self.n_cells - 1)
        self.ship_sunk = False
        self.probabilities = None
        print(f"⚓ Campo minato quantistico {grid_size}×{grid_size} = {self.n_cells} celle")
        print(f"🎛️  Qubit utilizzati: {self.n_qubits} (spazio di Hilbert: {2**self.n_qubits} stati)")
        print("⚠️  ATTENZIONE: simulazione estremamente pesante (richiede >4GB RAM e minuti di calcolo).")
        print("   Per una prova veloce riduci la griglia a 20×20 (400 celle, 9 qubit).\n")

    def index_to_coords(self, idx):
        return (idx // self.grid_size, idx % self.grid_size)

    def coords_to_index(self, r, c):
        return r * self.grid_size + c

    def grover_circuit(self, marked_idx, iterations):
        """Costruisce il circuito di Grover con matrici dense."""
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Sovrapposizione uniforme iniziale
        qc.h(range(n))

        # Matrice oracolo: -1 sullo stato marcato
        dim = 2**n
        print(f"   Creazione matrice oracolo ({dim}×{dim})...")
        oracle_matrix = np.eye(dim, dtype=complex)
        oracle_matrix[marked_idx, marked_idx] = -1
        oracle_gate = UnitaryGate(oracle_matrix, label='Oracle')

        # Matrice di diffusione: 2|s><s| - I
        print("   Creazione matrice diffusore...")
        s = np.ones(dim) / np.sqrt(dim)
        diff_matrix = 2 * np.outer(s, s) - np.eye(dim)
        diff_gate = UnitaryGate(diff_matrix, label='Diffuser')

        print(f"   Applicazione di {iterations} iterazioni...")
        for i in range(iterations):
            qc.append(oracle_gate, range(n))
            qc.append(diff_gate, range(n))
            if i % 5 == 0:
                print(f"   ... iterazione {i+1}/{iterations}")

        return qc

    def scan(self, iterations=2):
        """Esegue la ricerca e salva le probabilità."""
        if self.ship_sunk:
            print("Nave già affondata.")
            return

        print(f"\n🔍 Avvio scansione quantistica con {iterations} iterazioni...")
        start_time = time.time()

        backend = Aer.get_backend('statevector_simulator')
        qc = self.grover_circuit(self.ship_index, iterations)

        print("   Trasposizione circuito...")
        qc = transpile(qc, backend)

        print("   Esecuzione simulazione statevector...")
        job = backend.run(qc)
        statevector = job.result().get_statevector()

        # Probabilità per le prime n_cells celle
        probs_full = np.abs(statevector)**2
        self.probabilities = probs_full[:self.n_cells]

        elapsed = time.time() - start_time
        print(f"✅ Scansione completata in {elapsed:.1f} secondi.")

    def plot_probabilities(self):
        """Mostra il picco di probabilità e un grafico limitato."""
        if self.probabilities is None:
            print("Nessuna scansione eseguita.")
            return

        max_idx = np.argmax(self.probabilities)
        max_prob = self.probabilities[max_idx]
        print(f"\n📊 Probabilità massima: cella {self.index_to_coords(max_idx)} con {max_prob:.4f}")
        print(f"   Media probabilità: {np.mean(self.probabilities):.6f}")
        print(f"   La cella giusta dovrebbe essere quella con probabilità maggiore.")

        # Plot delle prime 200 celle (per dare un'idea, ma il picco potrebbe essere altrove)
        fig, ax = plt.subplots(figsize=(14, 5))
        n_show = min(200, self.n_cells)
        indices = range(n_show)
        labels = [f"{self.index_to_coords(i)}" for i in indices]
        ax.bar(indices, self.probabilities[:n_show], tick_label=labels)
        ax.set_xlabel('Cella (riga,colonna)')
        ax.set_ylabel('Probabilità')
        ax.set_title(f'Prime {n_show} celle - distribuzione probabilità')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def fire(self, row, col):
        """Spara sulla cella (r,c)."""
        idx = self.coords_to_index(row, col)
        if self.ship_sunk:
            print("Nave già affondata.")
            return True
        if idx == self.ship_index:
            print("💥 COLPO! Nave affondata!")
            self.ship_sunk = True
            return True
        else:
            print("💧 ACQUA. Continuare la ricerca.")
            return False

    def display_ascii(self):
        """Mostra solo un angolo della griglia (prime 5x5)."""
        size = min(5, self.grid_size)
        print(f"\n--- Angolo superiore sinistro della griglia {self.grid_size}×{self.grid_size} ---")
        print("   " + "   ".join(str(i) for i in range(size)))
        for r in range(size):
            row = str(r) + "  "
            for c in range(size):
                if self.ship_sunk and (r, c) == self.index_to_coords(self.ship_index):
                    row += "💥  "
                else:
                    row += "⬜  "
            print(row)
        print(f"... (resto della griglia non mostrato per dimensioni)\n")


def main():
    # Per test rapidi, riduci a 20 (400 celle) se il computer è lento
    game = QuantumBattleshipGrover(100)  # 100×100
    print("=== BATTAGLIA NAVALE QUANTISTICA GIGANTE ===")
    print("Obiettivo: trovare l'unica nave nascosta in 10.000 celle.")
    print("Usa la scansione quantistica per individuare la cella più probabile.\n")
    print("Comandi:")
    print("  scan [iter]  - avvia scansione (iterazioni, default 2, consigliato 2-3 per tempi ragionevoli)")
    print("  plot         - mostra probabilità (prime celle) e indica il picco")
    print("  fuoco r c    - spara sulla cella (r,c) con r,c da 0 a 99")
    print("  q            - esci\n")

    while True:
        game.display_ascii()
        cmd = input("> ").strip().split()
        if not cmd:
            continue
        if cmd[0].lower() == 'q':
            break
        elif cmd[0].lower() == 'scan':
            iters = 2
            if len(cmd) > 1:
                try:
                    iters = int(cmd[1])
                    if iters < 1 or iters > 10:
                        print("Iterazioni consigliate tra 1 e 5 per non esagerare.")
                except ValueError:
                    print("Iterazione non valida, uso 2")
            game.scan(iterations=iters)
        elif cmd[0].lower() == 'plot':
            game.plot_probabilities()
        elif cmd[0].lower() == 'fuoco':
            if len(cmd) != 3:
                print("Formato: fuoco r c")
                continue
            try:
                r, c = int(cmd[1]), int(cmd[2])
                if not (0 <= r < 100 and 0 <= c < 100):
                    print("Coordinate fuori griglia (0-99).")
                    continue
                game_over = game.fire(r, c)
                if game_over:
                    print("🎉 Vittoria! La flotta nemica è stata sconfitta.")
                    break
            except ValueError:
                print("Inserisci numeri interi.")
        else:
            print("Comando non riconosciuto.")

if __name__ == "__main__":
    main()
