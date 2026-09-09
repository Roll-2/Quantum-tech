import numpy as np
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate   # <-- import corretto

class QuantumBattleshipGrover:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.n_qubits = int(np.ceil(np.log2(self.n_cells)))  # 4 qubit (16 stati)
        self.ship_index = random.randint(0, self.n_cells - 1)  # indice della nave (0..8)
        self.ship_sunk = False
        self.probabilities = None

    def index_to_coords(self, idx):
        return (idx // self.grid_size, idx % self.grid_size)

    def coords_to_index(self, r, c):
        return r * self.grid_size + c

    def grover_circuit(self, marked_idx, iterations):
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Inizializzazione: H su tutti i qubit
        qc.h(range(n))

        # Matrice oracolo: diagonale con -1 sullo stato marcato
        dim = 2**n
        oracle_matrix = np.eye(dim, dtype=complex)
        oracle_matrix[marked_idx, marked_idx] = -1
        oracle_gate = UnitaryGate(oracle_matrix, label='Oracle')

        # Matrice di diffusione: 2|s><s| - I, dove |s> = H|0> (sovrapposizione uniforme)
        s = np.ones(dim) / np.sqrt(dim)
        diff_matrix = 2 * np.outer(s, s) - np.eye(dim)
        diff_gate = UnitaryGate(diff_matrix, label='Diffuser')

        # Applica le iterazioni di Grover
        for _ in range(iterations):
            qc.append(oracle_gate, range(n))
            qc.append(diff_gate, range(n))

        return qc

    def scan(self, iterations=2):
        if self.ship_sunk:
            print("La nave è già affondata. Non serve scansionare.")
            return

        print(f"\n🔍 Scansione quantistica con {iterations} iterazioni di Grover...")
        backend = Aer.get_backend('statevector_simulator')
        qc = self.grover_circuit(self.ship_index, iterations)
        qc = transpile(qc, backend)          # ottimizzazione opzionale
        job = backend.run(qc)
        statevector = job.result().get_statevector()

        # Probabilità per le prime n_cells celle (le altre 16-9 sono irrilevanti)
        probs_full = np.abs(statevector)**2
        self.probabilities = probs_full[:self.n_cells]
        print("Scansione completata. Usa 'plot' per vedere le probabilità.")

    def plot_probabilities(self):
        if self.probabilities is None:
            print("Nessuna scansione eseguita. Usa prima 'scan'.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        indices = range(self.n_cells)
        labels = [f"{self.index_to_coords(i)}" for i in indices]
        ax.bar(indices, self.probabilities, tick_label=labels)
        ax.set_xlabel('Cella (riga,colonna)')
        ax.set_ylabel('Probabilità')
        ax.set_title('Distribuzione di probabilità dopo scansione quantistica')
        plt.show()

    def fire(self, row, col):
        idx = self.coords_to_index(row, col)
        if self.ship_sunk:
            print("La nave è già stata affondata.")
            return True
        if idx == self.ship_index:
            print("💥 COLPO! La nave è stata colpita e affondata!")
            self.ship_sunk = True
            return True
        else:
            print("💧 ACQUA. La nave non è qui.")
            return False

    def display_ascii(self):
    # Intestazione: tre spazi iniziali, poi numeri separati da tre spazi
        header = "   " + "   ".join(str(i) for i in range(self.grid_size))
        print(header)
        for r in range(self.grid_size):
            # Riga: numero, due spazi, poi emoji separate da due spazi
            row = str(r) + "  "
            for c in range(self.grid_size):
                if self.ship_sunk and (r, c) == self.index_to_coords(self.ship_index):
                    row += "💥  "
                else:
                    row += "⬜  "
            print(row)
        print()

def main():
    game = QuantumBattleshipGrover(3)
    print("=== BATTAGLIA NAVALE QUANTISTICA (con Grover) ===")
    print("C'è UNA nave nascosta nella griglia 3x3.")
    print("Prima di sparare, esegui una scansione quantistica per localizzarla.")
    print("\nComandi:")
    print("  scan [iterazioni]  - esegue la ricerca di Grover (default 2)")
    print("  plot               - mostra il grafico delle probabilità")
    print("  fuoco r c          - spara sulla cella (r,c)")
    print("  q                  - esci\n")

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
                    print("Iterazioni non valide, uso 2")
            game.scan(iterations=iters)
        elif cmd[0].lower() == 'plot':
            game.plot_probabilities()
        elif cmd[0].lower() == 'fuoco':
            if len(cmd) != 3:
                print("Formato: fuoco r c")
                continue
            try:
                r, c = int(cmd[1]), int(cmd[2])
                if r < 0 or r >= game.grid_size or c < 0 or c >= game.grid_size:
                    print("Coordinate non valide.")
                    continue
                game_over = game.fire(r, c)
                if game_over:
                    game.display_ascii()
                    print("🎉 Partita terminata.")
                    break
            except ValueError:
                print("Inserisci numeri interi.")
        else:
            print("Comando non riconosciuto.")

if __name__ == "__main__":
    main()
