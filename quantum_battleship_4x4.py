import numpy as np
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate

class QuantumBattleshipGrover:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size          # 16
        self.n_qubits = int(np.ceil(np.log2(self.n_cells)))  # 4 qubit (16 stati)
        self.ship_index = random.randint(0, self.n_cells - 1)
        self.ship_sunk = False
        self.probabilities = None

    def index_to_coords(self, idx):
        return (idx // self.grid_size, idx % self.grid_size)

    def coords_to_index(self, r, c):
        return r * self.grid_size + c

    def grover_circuit(self, marked_idx, iterations):
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Sovrapposizione uniforme iniziale
        qc.h(range(n))

        # Matrice oracolo: -1 sullo stato marcato
        dim = 2**n
        oracle_matrix = np.eye(dim, dtype=complex)
        oracle_matrix[marked_idx, marked_idx] = -1
        oracle_gate = UnitaryGate(oracle_matrix, label='Oracle')

        # Matrice di diffusione: 2|s><s| - I
        s = np.ones(dim) / np.sqrt(dim)
        diff_matrix = 2 * np.outer(s, s) - np.eye(dim)
        diff_gate = UnitaryGate(diff_matrix, label='Diffuser')

        for _ in range(iterations):
            qc.append(oracle_gate, range(n))
            qc.append(diff_gate, range(n))

        return qc

    def scan(self, iterations=2):
        if self.ship_sunk:
            print("Nave già affondata.")
            return

        print(f"\n🔍 Scansione quantistica con {iterations} iterazioni di Grover...")
        backend = Aer.get_backend('statevector_simulator')
        qc = self.grover_circuit(self.ship_index, iterations)
        qc = transpile(qc, backend)
        job = backend.run(qc)
        statevector = job.result().get_statevector()

        probs_full = np.abs(statevector)**2
        self.probabilities = probs_full[:self.n_cells]
        qc.draw(output='mpl')
        print("Scansione completata. Usa 'plot' per vedere le probabilità.")


    def plot_probabilities(self):
        if self.probabilities is None:
            print("Nessuna scansione eseguita.")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        indices = range(self.n_cells)
        labels = [f"{self.index_to_coords(i)}" for i in indices]
        ax.bar(indices, self.probabilities, tick_label=labels)
        ax.set_xlabel('Cella (riga,colonna)')
        ax.set_ylabel('Probabilità')
        ax.set_title(f'Distribuzione probabilità {self.grid_size}x{self.grid_size}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def fire(self, row, col):
        idx = self.coords_to_index(row, col)
        if self.ship_sunk:
            print("Nave già affondata.")
            return True
        if idx == self.ship_index:
            print("💥 COLPO! Nave affondata!")
            self.ship_sunk = True
            return True
        else:
            print("💧 ACQUA.")
            return False

    def display_ascii(self):
        header = "   " + "   ".join(str(i) for i in range(self.grid_size))
        print(header)
        for r in range(self.grid_size):
            row = str(r) + "  "
            for c in range(self.grid_size):
                if self.ship_sunk and (r, c) == self.index_to_coords(self.ship_index):
                    row += "💥  "
                else:
                    row += "⬜  "
            print(row)
        print()


def main():
    game = QuantumBattleshipGrover(4)
    print("=== BATTAGLIA NAVALE QUANTISTICA 4x4 ===")
    print("Una nave nascosta in 16 celle.")
    print("Comandi: scan [iter], plot, fuoco r c, q\n")

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
                if not (0 <= r < 4 and 0 <= c < 4):
                    print("Coordinate fuori griglia.")
                    continue
                game_over = game.fire(r, c)
                if game_over:
                    game.display_ascii()
                    print("🎉 Partita finita.") 
                    break
            except ValueError:
                print("Inserisci numeri interi.")
        else:
            print("Comando sconosciuto.")

if __name__ == "__main__":
    main()
