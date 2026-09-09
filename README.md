# Quantum Playground — Battaglia Navale con Grover & Sfera di Bloch

Raccolta di programmi didattici in Python sul calcolo quantistico, divisi in due famiglie:

- **Tre versioni di Battaglia Navale quantistica** (`quantum_battleship_*.py`), che usano l'algoritmo di Grover (via Qiskit) per localizzare una nave nascosta.
- **Due visualizzatori interattivi della sfera di Bloch** (`bloch.py`, `bloch_circuit.py`), per esplorare stati e gate a un qubit.

## L'algoritmo di Grover in breve

Grover trova **un elemento marcato in un database non ordinato di N elementi in O(√N)** passi, contro gli O(N) richiesti classicamente. Con n qubit lo spazio di Hilbert ha N = 2ⁿ stati e l'algoritmo procede così:

1. **Sovrapposizione uniforme** — Hadamard su tutti i qubit: lo stato diventa |s⟩ = (1/√N) Σ|x⟩, ogni stato ha ampiezza 1/√N.
2. **Oracolo U_f** — inverte il segno dell'ampiezza dello stato marcato (phase kickback): |x⟩ → (−1)^f(x)|x⟩. Gli stati validi della griglia vengono "mappati" sui primi indici dello spazio.
3. **Diffusione (inversione rispetto alla media)** — l'operatore D = 2|s⟩⟨s| − I riflette tutte le ampiezze attorno alla loro media: l'ampiezza negativa dello stato marcato cresce in valore assoluto, quella degli altri cala. Geometricamente è una rotazione del vettore di stato nel piano generato da |s⟩ e |w⟩ (lo stato marcato).
4. **Iterazione** — si ripete il ciclo oracolo+diffusione R ≈ (π/4)·√N volte. La probabilità di misurare lo stato marcato è sin²((2R+1)θ) con sinθ = 1/√N: **troppi cicli fanno "superare" il picco** e la probabilità ricade.

Nelle battaglie navali l'oracolo e il diffusore sono costruiti come **matrici dense** (diag(−1) sull'indice della nave, e 2|s⟩⟨s|−I) applicate con `UnitaryGate`; la simulazione statevector di Aer restituisce ψ, e la probabilità di ogni cella è |ψ|². Per griglie non potenza di 2 (9 o 10.000 celle) il circuito lavora nello spazio più piccolo che le contiene (16 o 16.384 stati): gli stati extra restano a ampiezza zero e vengono ignorati nella lettura.

## I programmi

| File | Griglia | Celle | Qubit | Spazio di Hilbert |
|---|---|---|---|---|
| `quantum_battleship_3x3.py` | 3×3 | 9 | 4 | 16 stati |
| `quantum_battleship_4x4.py` | 4×4 | 16 | 4 | 16 stati |
| `quantum_battleship_100x100.py` | 100×100 | 10.000 | 14 | 16.384 stati |

### quantum_battleship_3x3.py / quantum_battleship_4x4.py

Nave nascosta casuale in una griglia; il giocatore esegue una scansione quantistica e spara dove la probabilità è massima. Con 9 celle bastano ~2 iterazioni (R ≈ π/4·3 ≈ 2.4), con 16 circa 3. La classe `QuantumBattleshipGrover` espone:

- `grover_circuit(marked_idx, iterations)` — Hadamard iniziali, oracolo e diffusore come `UnitaryGate`, ripetuti per il numero di iterazioni richiesto;
- `scan(iterations)` — traspila ed esegue il circuito sul simulatore statevector e salva le probabilità per cella;
- `plot_probabilities()` — grafico a barre della distribuzione di probabilità sulla griglia;
- `fire(r, c)` — colpo sulla cella: colpita o acqua;
- `display_ascii()` — griglia testuale con emoji (💥 quando la nave è affondata).

Comandi: `scan [iterazioni]`, `plot`, `fuoco r c`, `q`. La versione 4x4 disegna inoltre il circuito in matplotlib dopo ogni scansione.

### quantum_battleship_100x100.py

Versione "gigante": 10.000 celle, 14 qubit, matrici dense 16.384×16.384 per oracolo e diffusore. La simulazione è **molto pesante** (diversi GB di RAM e minuti di calcolo), quindi il programma stampa l'avanzamento passo per passo, mostra il tempo di scansione, indica la cella con probabilità massima e traccia solo le prime 200 celle; la griglia ASCII mostra soltanto l'angolo superiore sinistro. Per test rapidi è consigliabile ridurre la griglia (es. 20×20 = 400 celle, 9 qubit).

### bloch.py

Sfera di Bloch interattiva "manuale": lo stato |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩ è disegnato come freccia rossa sulla sfera, con etichetta in notazione bra-ket (coefficienti come frazioni note e angoli come multipli di π).

- **Mouse verticale** — cambia θ (dal polo |0⟩ al polo |1⟩);
- **A / D** — decrementa/incrementa φ di 15° (π/12);
- **X / Y / Z** — stati assiali |+⟩, |+i⟩, |0⟩; **R** — reset.

### bloch_circuit.py

Versione evoluta della sfera di Bloch, pensata per esplorare i gate:

- **Bottoni** per i gate I, X, Y, Z, H, S, T, S†, T†: ognuno applica la sua matrice 2×2 allo stato corrente;
- **Frecce ↑/↓** per θ e **A/D** per φ, con stati rapidi X/Y/Z e reset con **R**;
- **Play** — anima la sequenza H → S → T: ogni gate è interpretato come rotazione del vettore di Bloch attorno al proprio asse (es. H ruota di π attorno all'asse (x+z)/√2), mostrata in 30 passi da 50 ms.

Le funzioni chiave sono `stato_da_angoli` / `angoli_da_stato` (conversione bidirezionale tra vettore di stato e angoli di Bloch) e `ruota_vettore` (rotazione di Rodrigues per l'animazione).

## Requisiti

```bash
pip install numpy matplotlib qiskit qiskit-aer
```

## Uso

```bash
python quantum_battleship_3x3.py     # o 4x4 / 100x100
python bloch.py
python bloch_circuit.py
```

I programmi richiedono un desktop con finestre matplotlib (nessun output headless).
